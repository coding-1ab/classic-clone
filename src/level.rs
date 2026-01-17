use glam::{IVec3, Vec2, Vec3};

struct LevelData {
    palette: Vec<Identifier>,
    chunks: [Chunk; 4096],
    player_data: PlayerData,
}

struct Chunk {
    position: IVec3,
    data: [i32; 4096],
}

type SlotSize = u8;

pub struct PlayerData {
    pub position: Vec3,
    facing: Vec2,
    selected_slot: SlotSize,
    inventory: [Slot; 9],
}

struct Slot {
    slot: SlotSize,
    item: Item,
}

struct Item {
    id: Identifier,
    count: u8,
}

struct Identifier {
    value: String,
}
