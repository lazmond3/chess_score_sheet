pub trait ChessClub {
    fn name(&self) -> &String;
    fn url(&self) -> &String;
    fn scrape_event(&self) -> String;
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
    fn scrape_event(&self) -> String {
        let body = reqwest::blocking::get(self.url()).unwrap().text().unwrap();
        let document = scraper::Html::parse_document(&body);

        let selector = scraper::Selector::parse("div.entry-content p").unwrap();
        let elements = document.select(&selector);

        let mut ret = String::from("");
        for e in elements {
            let text = e.text().next().unwrap();
            ret += text;
        }

        ret
    }

    // fn new(name, url) -> ChessClub8x8 {name=};
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    println!("{:?}", args);
    let mut target = "8x8"; // TODO: change to "ALL"
    if args.len() > 1 {
        target = &args[1];
    }
    println!("target: {:?}", target);

    let target_club = create_chess_club(target);

    println!("target_club.name: {:?}", target_club.name());
    println!("target_club.url: {:?}", target_club.url());
    println!("target_club.scrape_event: {:?}", target_club.scrape_event());
}

fn create_chess_club(target: &str) -> Box<dyn ChessClub> {
    let target_club;

    match target {
        "8x8" => {
            target_club = ChessClub8x8 {
                _name: String::from(target),
                _url: String::from("https://8by8.hatenablog.com/"),
            };
        }
        _ => panic!("Error, not supported target: ${:?}", target),
    }

    Box::new(target_club)
}

// fn hello_scraper() -> Result<(), Box<dyn std::error::Error>> {
// const CHESS_HP_LIST: [&str; 1] = ["http://chess.m1.valueserver.jp/"];
//     // セレクターをパース　(このセレクターは記事のアンカーノード群(タイトル)を指す。 <a href="link">Title</a>)
//     // let selector = scraper::Selector::parse("td.bn > a").unwrap();
//     let selector = scraper::Selector::parse("div.item h4").unwrap();

//     // `https://blog.rust-lang.org/` へHTTPリクエスト
//     // let body = reqwest::blocking::get("https://blog.rust-lang.org/")?.text()?;
//     let body = reqwest::blocking::get(CHESS_HP_LIST[0])?.text()?;

//     // HTMLをパース
//     let document = scraper::Html::parse_document(&body);

//     // セレクターを用いて要素を取得
//     let elements = document.select(&selector);

//     // 全記事名を出力
//     // elements.for_each(|e| println!("{}", e.text().next().unwrap()));
//     for e in elements {
//         // e.what_is_this();
//         let text = e.text().next().unwrap();
//         // e.text().next().what_is_this();
//         // e.text().next().unwrap().what_is_this();
//         // let (res, _, _) = encoding_rs::SHIFT_JIS.decode(text);
//         // let text = res.into_owned();
//         println!("{}", text);
//         break;
//     }

//     Ok(())
// }

// fn get_8by8_chess() -> Result<String, Box<dyn std::error::Error>> {
// }
