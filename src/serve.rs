use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Result, anyhow};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use burn::module::Module;
use log::info;
use serde::{Deserialize, Serialize};
use tokio::net::TcpListener;
use tokio::runtime::Runtime;
use tokio::sync::Mutex;
use tokio::task::block_in_place;

use crate::backend::{Device, Inference};
use crate::config::model::ModelConfig;
use crate::generate;
use crate::model::Phisch;
use crate::sampling::{MAX_CONTEXT_DEFAULT, Sampling};
use crate::store::load_model;
use crate::tokenizer::{ASSISTANT, END, SYSTEM, Tokenizer, USER};

struct Loaded {
    model: Phisch<Inference>,
    tokenizer: Tokenizer,
    config: ModelConfig,
    max_context: usize,
}

#[derive(Clone)]
struct AppState {
    directory: PathBuf,
    loaded: Arc<Mutex<Option<Loaded>>>,
    status: Arc<Mutex<Option<ModelStatus>>>,
}

pub fn serve(host: &str, port: u16, directory: PathBuf) -> Result<()> {
    Runtime::new()?.block_on(run(host, port, directory))
}

async fn run(host: &str, port: u16, directory: PathBuf) -> Result<()> {
    let state = AppState {
        directory,
        loaded: Arc::default(),
        status: Arc::default(),
    };

    let app = Router::new()
        .route("/", get(String::from("Phisch is listening!")))
        .route("/status", get(status))
        .route("/load", post(load))
        .route("/unload", post(unload))
        .route("/complete", post(complete))
        .route("/chat", post(chat))
        .with_state(state);

    let listener = TcpListener::bind((host, port)).await?;
    info!("Serving on {}", listener.local_addr()?);
    axum::serve(listener, app).await?;

    Ok(())
}

#[derive(Clone, Serialize)]
struct ModelStatus {
    checkpoint: String,
    max_context: usize,
    parameters: usize,
    vocab: usize,
    config: ModelConfig,
}

#[derive(Serialize)]
struct StatusResponse {
    directory: PathBuf,
    model: Option<ModelStatus>,
}

#[derive(Deserialize)]
struct LoadRequest {
    checkpoint: Option<String>,
    max_context: Option<usize>,
}

#[derive(Deserialize)]
struct CompleteRequest {
    prompt: String,
    #[serde(default)]
    sampling: Sampling,
}

#[derive(Serialize)]
struct CompleteResponse {
    text: String,
}

#[derive(Serialize, Deserialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct ChatRequest {
    messages: Vec<Message>,
    #[serde(default)]
    sampling: Sampling,
}

#[derive(Serialize)]
struct ChatResponse {
    message: Message,
}

async fn load(
    State(state): State<AppState>,
    Json(request): Json<LoadRequest>,
) -> Result<StatusCode, AppError> {
    info!("POST /load");
    let max_context = request.max_context.unwrap_or(MAX_CONTEXT_DEFAULT);

    let mut loaded = state
        .loaded
        .try_lock()
        .map_err(|_| anyhow!("model is busy, try again later"))?;
    let mut status = state.status.lock().await;

    *loaded = None;
    *status = None;

    let (model, tokenizer, config, checkpoint) = block_in_place(|| {
        load_model(
            &state.directory,
            request.checkpoint.as_deref(),
            max_context,
            &Device::default(),
        )
    })?;

    *status = Some(ModelStatus {
        checkpoint,
        max_context,
        parameters: model.num_params(),
        vocab: tokenizer.vocab_size(),
        config: config.clone(),
    });

    *loaded = Some(Loaded {
        model,
        tokenizer,
        config,
        max_context,
    });

    Ok(StatusCode::NO_CONTENT)
}

async fn unload(State(state): State<AppState>) -> Result<StatusCode, AppError> {
    info!("POST /unload");
    state
        .loaded
        .try_lock()
        .map_err(|_| anyhow!("model is busy, try again later"))?
        .take();
    state.status.lock().await.take();

    Ok(StatusCode::NO_CONTENT)
}

async fn status(State(state): State<AppState>) -> Result<Json<StatusResponse>, AppError> {
    info!("GET /status");
    let status = state.status.lock().await;
    Ok(Json(StatusResponse {
        directory: state.directory.clone(),
        model: status.clone(),
    }))
}

async fn complete(
    State(state): State<AppState>,
    Json(request): Json<CompleteRequest>,
) -> Result<Json<CompleteResponse>, AppError> {
    info!("POST /complete");
    let text = with_model(&state, move |loaded| {
        generate_text(loaded, &request.prompt, &request.sampling, false)
    })
    .await?;

    Ok(Json(CompleteResponse { text }))
}

async fn chat(
    State(state): State<AppState>,
    Json(request): Json<ChatRequest>,
) -> Result<Json<ChatResponse>, AppError> {
    info!("POST /chat");
    let prompt = chat_prompt(&request.messages)?;
    let content = with_model(&state, move |loaded| {
        generate_text(loaded, &prompt, &request.sampling, true)
    })
    .await?;

    Ok(Json(ChatResponse {
        message: Message {
            role: "assistant".to_string(),
            content,
        },
    }))
}

fn chat_prompt(messages: &[Message]) -> Result<String> {
    let turns = messages
        .iter()
        .map(|message| {
            let tag = match message.role.as_str() {
                "system" => SYSTEM,
                "user" => USER,
                "assistant" => ASSISTANT,
                role => anyhow::bail!("unknown role {role}, use system, user or assistant"),
            };

            Ok(format!("{tag}{}{END}", message.content))
        })
        .collect::<Result<String>>()?;

    Ok(format!("{turns}{ASSISTANT}"))
}

fn generate_text(
    loaded: &Loaded,
    prompt: &str,
    sampling: &Sampling,
    skip_special_tokens: bool,
) -> Result<String> {
    anyhow::ensure!(
        sampling.max_context <= loaded.max_context,
        "max_context ({}) can't be larger than the one the model was loaded with ({})",
        sampling.max_context,
        loaded.max_context
    );

    let ids = loaded.tokenizer.encode(prompt)?;
    let end = loaded.tokenizer.id(END)?;
    let generated = generate::complete(
        &loaded.model,
        &loaded.config,
        &loaded.tokenizer,
        &ids,
        end,
        sampling,
    )?;

    loaded.tokenizer.decode(&generated, skip_special_tokens)
}

async fn with_model<T, F>(state: &AppState, f: F) -> Result<T, AppError>
where
    F: FnOnce(&Loaded) -> anyhow::Result<T>,
{
    let loaded = state.loaded.lock().await;
    let loaded = loaded.as_ref().ok_or_else(|| anyhow!("no model loaded"))?;

    Ok(block_in_place(|| f(loaded))?)
}

struct AppError(anyhow::Error);

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        (StatusCode::INTERNAL_SERVER_ERROR, self.0.to_string()).into_response()
    }
}

impl<E: Into<anyhow::Error>> From<E> for AppError {
    fn from(err: E) -> Self {
        Self(err.into())
    }
}
