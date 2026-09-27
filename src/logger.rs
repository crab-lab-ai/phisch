use log::{Level, Metadata, Record};

pub struct Logger;

impl log::Log for Logger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        metadata.level() <= Level::Info
    }

    fn log(&self, record: &Record) {
        if self.enabled(record.metadata()) {
            println!("\x1b[0;32m>\x1b[0m {}", record.args());
        }
    }

    fn flush(&self) {}
}
