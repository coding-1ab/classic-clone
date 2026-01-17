use crate::player::Player;

pub trait Tickable {
    fn tick(&mut self);
}

struct Game {
    player: Player,
}

impl Tickable for Game {
    fn tick(&mut self) {
        self.player.tick();
    }
}
