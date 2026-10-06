use colored::Colorize;
use log::{Level, Metadata, Record};

pub struct Logger;

impl log::Log for Logger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        let is_phisch = metadata.target().strip_prefix("phisch").is_some();
        metadata.level() <= Level::Info && is_phisch
    }

    fn log(&self, record: &Record) {
        if self.enabled(record.metadata()) {
            let dingle = ">";
            let dingle = match record.metadata().level() {
                Level::Error => dingle.red(),
                Level::Warn => dingle.yellow(),
                Level::Info => dingle.blue(),
                Level::Debug => dingle.magenta(),
                Level::Trace => dingle.bright_magenta(),
            };
            println!("{} {}", dingle, record.args());
        }
    }

    fn flush(&self) {}
}
