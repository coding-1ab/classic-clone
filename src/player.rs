use crate::game::Tickable;
use crate::level::PlayerData;
use glam::Vec3;

static SLIPPERINESS: f32 = 0.3;

pub struct Player {
    data: PlayerData,
    velocity: Vec3,
}

impl Player {
    fn tick_movement(&mut self) {
        if self.velocity.length() < 0.00005 {
            self.velocity = Vec3::ZERO;
        } else {
            self.tick_movement_input();
            self.velocity *= SLIPPERINESS;
            self.data.position += self.velocity;
        }
    }

    fn tick_movement_input(&mut self) {}
}

impl Tickable for Player {
    fn tick(&mut self) {
        self.tick_movement();
    }
}
