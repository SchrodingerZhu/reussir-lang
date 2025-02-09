use ariadne::{Color, Label, Report, ReportKind, Source};

use crate::syntax::surface::Msg;

mod syntax;

#[allow(dead_code)]
fn print_parse_errors<'src>(es: &[Msg<'src>], src: &'src str) {
    es.iter().for_each(|e| {
        let range = e.span();
        let src_id = "<stdin>";
        Report::build(ReportKind::Error, (src_id, range.start..range.start))
            .with_message("parse error")
            .with_label(
                Label::new((src_id, range.into_range()))
                    .with_message(e.clone().map_token(char::escape_default).reason())
                    .with_color(Color::Red),
            )
            .finish()
            .eprint((src_id, Source::from(src)))
            .unwrap();
    })
}
