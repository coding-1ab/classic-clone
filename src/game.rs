pub trait Tickable {
    fn tick(&mut self);
}

struct Game {}

impl Tickable for Game {
    fn tick(&mut self) {
        todo!()
    }
}
