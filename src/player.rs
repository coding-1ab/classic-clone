use crate::game::Tickable;
use crate::level::PlayerData;

pub struct Player {
    data: PlayerData,
}

impl Tickable for Player {
    fn tick(&mut self) {}
}
