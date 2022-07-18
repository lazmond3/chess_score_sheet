pub struct EventInfo {
    date: String,
    open_time: String,
    revenue: String,
    fee: String,
}

pub trait ChessClub {
    fn name(&self) -> &String;
    fn url(&self) -> &String;
    fn scrape_event(&self) -> Vec<EventInfo>;
}

struct ChessClub8x8 {
    _name: String,
    _url: String,
}

impl ChessClub for ChessClub8x8 {
    fn name(&self) -> &String {
        &self._name
    }
    fn url(&self) -> &String {
        &self._url
    }
    fn scrape_event(&self) -> Vec<EventInfo> {
        let body = reqwest::blocking::get(self.url()).unwrap().text().unwrap();
        let document = scraper::Html::parse_document(&body);

        let div_selector = scraper::Selector::parse("div.entry-content").unwrap();
        let div_elements = document.select(&div_selector);
        let p_selector = scraper::Selector::parse("p").unwrap();

        let mut events = Vec::new();
        for div in div_elements {
            log::info!("================");
            let mut date = String::from("");
            let mut open_time = String::from("");
            let mut revenue = String::from("");
            let mut fee = String::from("");
            for e in div.select(&p_selector) {
                let text = e.text().collect::<Vec<_>>().join("");
                log::info!("text: {:?}", text);

                if text.contains("場所:") {
                    revenue = String::from(text.trim().trim_start_matches("場所:").trim());
                }
                if text.contains("日時:") {
                    date = String::from(text.trim().trim_start_matches("日時:").trim());
                    date = trim_left(
                        &date,
                        Vec::from([String::from("(定員"), String::from("（定員")]),
                    );
                }
                if text.contains("参加費:") {
                    fee = String::from(text.trim().trim_start_matches("参加費:").trim());
                }
                let re = regex::Regex::new(r"(\d{2})時\d{2}分〜\d{2}時\d{2}分").unwrap();
                if re.is_match(&text) {
                    open_time = String::from(text.trim());
                }
            }

            let e = EventInfo {
                date,
                open_time,
                revenue,
                fee,
            };
            events.push(e);
        }

        events
    }
}

struct ChessClubKitaSenjyu {
    _name: String,
    _url: String,
}

impl ChessClub for ChessClubKitaSenjyu {
    fn name(&self) -> &String {
        &self._name
    }
    fn url(&self) -> &String {
        &self._url
    }
    fn scrape_event(&self) -> Vec<EventInfo> {
        let body = reqwest::blocking::get(self.url()).unwrap().text().unwrap();
        let document = scraper::Html::parse_document(&body);

        let div_selector = scraper::Selector::parse("div.entry-content").unwrap();
        let div_elements = document.select(&div_selector);
        let p_selector = scraper::Selector::parse("p").unwrap();

        let mut events = Vec::new();
        for div in div_elements {
            // log::info!("================");
            let mut date = String::from("");
            let mut open_time = String::from("");
            let mut revenue = String::from("");
            let mut fee = String::from("");
            for e in div.select(&p_selector) {
                let text = e.text().collect::<Vec<_>>().join("");
                // log::info!("text: {:?}", text);

                if text.contains("場所:") {
                    revenue = String::from(text.trim().trim_start_matches("場所:").trim());
                }
                if text.contains("日時:") {
                    date = String::from(text.trim().trim_start_matches("日時:").trim());
                    date = trim_left(
                        &date,
                        Vec::from([String::from("(定員"), String::from("（定員")]),
                    );
                }
                if text.contains("参加費:") {
                    fee = String::from(text.trim().trim_start_matches("参加費:").trim());
                }
                let re = regex::Regex::new(r"(\d{2})時\d{2}分〜\d{2}時\d{2}分").unwrap();
                if re.is_match(&text) {
                    open_time = String::from(text.trim());
                    // log::info!("open_time: {:?}", open_time);
                } else {
                    // log::info!("NOT open_time: {:?}", text);
                }
            }

            let e = EventInfo {
                date,
                open_time,
                revenue,
                fee,
            };
            events.push(e);
        }

        events
    }
}

fn main() {
    env_logger::init();

    let args: Vec<String> = std::env::args().collect();
    log::info!("args: {:?}", args);
    let mut target = "8x8"; // TODO: change to "ALL"
    if args.len() > 1 {
        target = &args[1];
    }
    log::info!("target: {:?}", target);

    let target_club = create_chess_club(target);

    log::info!("target_club.name: {:?}", target_club.name());
    log::info!("target_club.url: {:?}", target_club.url());
    for e in target_club.scrape_event() {
        log::info!("========");
        log::info!("date: {:?}", e.date);
        log::info!("open_time: {:?}", e.open_time);
        log::info!("revenue: {:?}", e.revenue);
        log::info!("fee: {:?}", e.fee);
    }
}

fn create_chess_club(target: &str) -> Box<dyn ChessClub> {
    let target_club: Box<dyn ChessClub> = match target {
        "8x8" => Box::new(ChessClub8x8 {
            _name: String::from(target),
            _url: String::from("https://8by8.hatenablog.com/"),
        }),
        "KitaSenjyu" => Box::new(ChessClubKitaSenjyu {
            _name: String::from(target),
            _url: String::from("https://blog.rust-lang.org/"),
        }),
        _ => panic!("Error, not supported target: ${:?}", target),
    };

    target_club
}

fn trim_left(text: &str, patterns: Vec<String>) -> String {
    let mut ret = text;
    for p in patterns {
        ret = match ret.find(&p) {
            Some(val) => ret[..val].trim(),
            None => ret,
        };
    }

    String::from(ret)
}
