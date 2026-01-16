extern crate glfw;
extern crate gl;

use glfw::{Action, Context, Key};
use glam::{Mat4, Vec3};
use std::mem;
use std::ptr;
use std::ffi::CString;

// --- 카메라 상태 ---
struct Camera {
    position: Vec3,
    yaw: f32,   // Y축 회전 (좌우 시점) - 라디안
    pitch: f32, // X축 회전 (상하 시점) - 라디안
    move_speed: f32,
    rotate_speed: f32,
}

impl Camera {
    fn new() -> Self {
        Camera {
            position: Vec3::new(0.0, 0.0, 2.5), // 초기 카메라 위치 (물체 앞)
            yaw: 0.0,
            pitch: 0.0,
            move_speed: 0.05,
            rotate_speed: 0.03, // 회전 속도 (라디안)
        }
    }

    fn move_forward(&mut self) {
        self.position.z -= self.move_speed;
    }

    fn move_backward(&mut self) {
        self.position.z += self.move_speed;
    }

    fn move_left(&mut self) {
        self.position.x -= self.move_speed;
    }

    fn move_right(&mut self) {
        self.position.x += self.move_speed;
    }

    fn move_up(&mut self) {
        self.position.y += self.move_speed;
    }

    fn move_down(&mut self) {
        self.position.y -= self.move_speed;
    }

    // 시점 회전 (WASD)
    fn look_up(&mut self) {
        self.pitch += self.rotate_speed;
        // 시점 제한 (90도 이상 넘어가지 않도록)
        self.pitch = self.pitch.clamp(-std::f32::consts::FRAC_PI_2 + 0.1, std::f32::consts::FRAC_PI_2 - 0.1);
    }

    fn look_down(&mut self) {
        self.pitch -= self.rotate_speed;
        self.pitch = self.pitch.clamp(-std::f32::consts::FRAC_PI_2 + 0.1, std::f32::consts::FRAC_PI_2 - 0.1);
    }

    fn look_left(&mut self) {
        self.yaw -= self.rotate_speed;
    }

    fn look_right(&mut self) {
        self.yaw += self.rotate_speed;
    }

    fn get_view_matrix(&self) -> Mat4 {
        // 뷰 행렬: 회전 후 이동 (카메라 변환의 역변환)
        // 순서: 이동의 역 -> pitch 회전의 역 -> yaw 회전의 역
        let translation = Mat4::from_translation(-self.position);
        let rotation = Mat4::from_rotation_x(-self.pitch) * Mat4::from_rotation_y(self.yaw);
        rotation * translation
    }
    
    // 카메라가 바라보는 방향 벡터 반환
    fn get_forward_direction(&self) -> Vec3 {
        // 기본 방향은 -Z (OpenGL 관례)
        // yaw와 pitch를 적용하여 실제 바라보는 방향 계산
        let x = self.yaw.sin() * self.pitch.cos();
        let y = self.pitch.sin();
        let z = -self.yaw.cos() * self.pitch.cos();
        Vec3::new(x, y, z).normalize()
    }
}

// --- 상수 정의 ---
const GRID_SIZE: usize = 5;
const BLOCK_SIZE: f32 = 0.2; // 1.0 / 5.0
const WORLD_ORIGIN: f32 = -0.5; // 시작점 (중앙 정렬을 위해)

// 방향 인덱스 (문제에서 제시한 순서: +x, -x, +y, -y, +z, -z)
const FACE_RIGHT: usize = 0;  // +x
const FACE_LEFT: usize = 1;   // -x
const FACE_TOP: usize = 2;    // +y
const FACE_BOTTOM: usize = 3; // -y
const FACE_FRONT: usize = 4;  // +z
const FACE_BACK: usize = 5;   // -z

// 쉐이더 소스 (이전과 동일)
const VERTEX_SHADER_SRC: &str = r#"
    #version 330 core
    layout (location = 0) in vec3 aPos;
    layout (location = 1) in vec3 aColor;

    out vec3 ourColor;
    
    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;

    void main() {
        gl_Position = projection * view * model * vec4(aPos, 1.0);
        ourColor = aColor;
    }
"#;

const FRAGMENT_SHADER_SRC: &str = r#"
    #version 330 core
    in vec3 ourColor;
    out vec4 FragColor;

    void main() {
        FragColor = vec4(ourColor, 1.0);
    }
"#;

// 크로스헤어용 셰이더 (2D)
const CROSSHAIR_VERTEX_SHADER_SRC: &str = r#"
    #version 330 core
    layout (location = 0) in vec2 aPos;

    void main() {
        gl_Position = vec4(aPos, 0.0, 1.0);
    }
"#;

const CROSSHAIR_FRAGMENT_SHADER_SRC: &str = r#"
    #version 330 core
    out vec4 FragColor;

    void main() {
        FragColor = vec4(1.0, 1.0, 1.0, 1.0);
    }
"#;

fn main() {
    // 1. GLFW 초기화
    let mut glfw = glfw::init(glfw::fail_on_errors).unwrap();
    glfw.window_hint(glfw::WindowHint::ContextVersion(3, 3));
    glfw.window_hint(glfw::WindowHint::OpenGlProfile(glfw::OpenGlProfileHint::Core));
    #[cfg(target_os = "macos")]
    glfw.window_hint(glfw::WindowHint::OpenGlForwardCompat(true));

    let (mut window, events) = glfw.create_window(240, 180, "Voxel 5x5x5 Engine", glfw::WindowMode::Windowed)
        .expect("Failed to create GLFW window.");

    window.make_current();
    window.set_key_polling(true);

    // 카메라 초기화
    let mut camera = Camera::new();

    // 2. GL 로딩
    gl::load_with(|s| {
        window.get_proc_address(s)
            .map(|ptr| ptr as *const _)   // 값이 있으면 포인터로 변환
            .unwrap_or(std::ptr::null())  // 값이 없으면(None) null 포인터 반환
    });

    // 3. 데이터 생성 (월드 데이터 & 팔레트)
    let (world_data, palette) = init_data();
    
    // 선택된 블록 추적 (이전 프레임에서 선택된 블록)
    let mut selected_block: Option<(usize, usize, usize)> = None;

    let (mut vbo, mut vao) = (0, 0);
    let (mut crosshair_vbo, mut crosshair_vao) = (0, 0);
    let shader_program;
    let crosshair_shader_program;
    
    // 크로스헤어 정점 데이터 (화면 중앙에 + 모양)
    let crosshair_size = 0.03; // NDC 좌표 기준 크기
    let crosshair_vertices: [f32; 8] = [
        // 수평선
        -crosshair_size, 0.0,
        crosshair_size, 0.0,
        // 수직선
        0.0, -crosshair_size,
        0.0, crosshair_size,
    ];
    
    unsafe {
        gl::Enable(gl::DEPTH_TEST); // 깊이 테스트 켜기 (앞이 뒤를 가리도록)

        shader_program = create_shader_program();
        crosshair_shader_program = create_crosshair_shader_program();

        // 메인 복셀 VAO/VBO
        gl::GenVertexArrays(1, &mut vao);
        gl::GenBuffers(1, &mut vbo);

        gl::BindVertexArray(vao);
        gl::BindBuffer(gl::ARRAY_BUFFER, vbo);

        let stride = (6 * mem::size_of::<f32>()) as i32;
        gl::VertexAttribPointer(0, 3, gl::FLOAT, gl::FALSE, stride, ptr::null()); // Pos
        gl::EnableVertexAttribArray(0);
        gl::VertexAttribPointer(1, 3, gl::FLOAT, gl::FALSE, stride, (3 * mem::size_of::<f32>()) as *const _); // Color
        gl::EnableVertexAttribArray(1);
        
        // 크로스헤어 VAO/VBO
        gl::GenVertexArrays(1, &mut crosshair_vao);
        gl::GenBuffers(1, &mut crosshair_vbo);
        
        gl::BindVertexArray(crosshair_vao);
        gl::BindBuffer(gl::ARRAY_BUFFER, crosshair_vbo);
        gl::BufferData(
            gl::ARRAY_BUFFER,
            (crosshair_vertices.len() * mem::size_of::<f32>()) as isize,
            crosshair_vertices.as_ptr() as *const _,
            gl::STATIC_DRAW,
        );
        
        gl::VertexAttribPointer(0, 2, gl::FLOAT, gl::FALSE, (2 * mem::size_of::<f32>()) as i32, ptr::null());
        gl::EnableVertexAttribArray(0);
    }

    // 6. 렌더링 루프
    println!("=== 조작법 ===");
    println!("[이동]");
    println!("  ↑: 앞으로 이동");
    println!("  ↓: 뒤로 이동");
    println!("  ←: 왼쪽으로 이동");
    println!("  →: 오른쪽으로 이동");
    println!("  Space: 위로 이동 (Y+)");
    println!("  Shift: 아래로 이동 (Y-)");
    println!("[시점 회전]");
    println!("  W: 위쪽 보기");
    println!("  S: 아래쪽 보기");
    println!("  A: 왼쪽 보기");
    println!("  D: 오른쪽 보기");
    println!("[종료]");
    println!("  ESC: 종료");

    while !window.should_close() {
        for (_, event) in glfw::flush_messages(&events) {
            match event {
                glfw::WindowEvent::Key(Key::Escape, _, Action::Press, _) => {
                    window.set_should_close(true);
                }
                // 방향키 - 앞/뒤/좌/우 이동
                glfw::WindowEvent::Key(Key::Up, _, Action::Press | Action::Repeat, _) => {
                    camera.move_forward();
                }
                glfw::WindowEvent::Key(Key::Down, _, Action::Press | Action::Repeat, _) => {
                    camera.move_backward();
                }
                glfw::WindowEvent::Key(Key::Left, _, Action::Press | Action::Repeat, _) => {
                    camera.move_left();
                }
                glfw::WindowEvent::Key(Key::Right, _, Action::Press | Action::Repeat, _) => {
                    camera.move_right();
                }
                // 스페이스바 - 위로 이동 (Y+)
                glfw::WindowEvent::Key(Key::Space, _, Action::Press | Action::Repeat, _) => {
                    camera.move_up();
                }
                // 쉬프트 - 아래로 이동 (Y-)
                glfw::WindowEvent::Key(Key::LeftShift, _, Action::Press | Action::Repeat, _) |
                glfw::WindowEvent::Key(Key::RightShift, _, Action::Press | Action::Repeat, _) => {
                    camera.move_down();
                }
                // WASD - 시점 회전
                glfw::WindowEvent::Key(Key::W, _, Action::Press | Action::Repeat, _) => {
                    camera.look_up();
                }
                glfw::WindowEvent::Key(Key::S, _, Action::Press | Action::Repeat, _) => {
                    camera.look_down();
                }
                glfw::WindowEvent::Key(Key::A, _, Action::Press | Action::Repeat, _) => {
                    camera.look_left();
                }
                glfw::WindowEvent::Key(Key::D, _, Action::Press | Action::Repeat, _) => {
                    camera.look_right();
                }
                _ => {}
            }
        }
        
        // 레이캐스팅으로 현재 바라보는 블록 찾기
        let new_selected = raycast_block(&camera, &world_data, 10.0);
        
        // 선택된 블록이 변경되었을 때만 메시 재생성
        if new_selected != selected_block {
            selected_block = new_selected;
        }
        
        // 메시 생성 (선택된 블록은 팔레트 5번으로 표시)
        let vertices = generate_mesh_with_selection(&world_data, &palette, selected_block);

        unsafe {
            gl::ClearColor(0.1, 0.1, 0.1, 1.0);
            gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);

            // --- 복셀 렌더링 ---
            gl::UseProgram(shader_program);

            // Model: 고정 (회전 없음)
            let model = Mat4::IDENTITY;
            
            // View: 카메라 위치에 따른 뷰 행렬
            let view = camera.get_view_matrix();
            
            // Projection
            let projection = Mat4::perspective_rh_gl(45.0f32.to_radians(), 240.0 / 180.0, 0.1, 100.0);

            set_mat4(shader_program, "model", &model);
            set_mat4(shader_program, "view", &view);
            set_mat4(shader_program, "projection", &projection);

            // 메시 데이터 업데이트
            gl::BindVertexArray(vao);
            gl::BindBuffer(gl::ARRAY_BUFFER, vbo);
            gl::BufferData(
                gl::ARRAY_BUFFER,
                (vertices.len() * mem::size_of::<f32>()) as isize,
                vertices.as_ptr() as *const _,
                gl::DYNAMIC_DRAW,
            );
            
            // 삼각형 개수 = 전체 float 개수 / 6(속성수)
            gl::DrawArrays(gl::TRIANGLES, 0, (vertices.len() / 6) as i32);
            
            // --- 크로스헤어 렌더링 ---
            gl::Disable(gl::DEPTH_TEST); // 크로스헤어는 항상 맨 앞에
            gl::UseProgram(crosshair_shader_program);
            gl::BindVertexArray(crosshair_vao);
            gl::DrawArrays(gl::LINES, 0, 4); // 2개의 선 (4개 정점)
            gl::Enable(gl::DEPTH_TEST);
        }

        window.swap_buffers();
        glfw.poll_events();
    }
}

// --- 데이터 초기화 함수 ---
#[allow(dead_code)]
fn init_data() -> ([[[u8; GRID_SIZE]; GRID_SIZE]; GRID_SIZE], [[[f32; 3]; 6]; 6]) {
    // 1. 월드 데이터 초기화 (임의의 모양 생성)
    let mut world = [[[0u8; GRID_SIZE]; GRID_SIZE]; GRID_SIZE];

    // 간단히 만들어본 집 모양 월드

    for y in 0..GRID_SIZE {
        for x in 0..GRID_SIZE {
            for z in 0..GRID_SIZE {
                if y == 0 {
                    world[x][y][z] = 2;
                } else if y < 3 {
                    if (1 <= x && x <= 3) && (1 <= z && z <= 3) {
                        world[x][y][z] = 3;
                    }
                    if (x == 2) && (z == 3) {
                        world[x][y][z] = 4;
                    }
                }
                else if y==3 {
                    if (1 <= x && x <= 3) && (1 <= z && z <= 3) {
                        world[x][y][z] = 1;
                    }
                }
                else {
                    if (x == 2) && (z == 2) {
                        world[x][y][z] = 1;
                    }
                }
            }
        }
    }

    // 2. 팔레트 정의 (블록 ID별 6면 색상)
    // [ID][FACE][RGB]
    // 0번은 공기이므로 정의는 하지만 쓰이지 않음.
    let mut palette = [[[0.0; 3]; 6]; 6];

    // 색상 생성 헬퍼
    let make_color = |r, g, b| [r, g, b];

    // ID 1: 붉은 계열 (기둥)
    palette[1] = [
        make_color(1.0, 0.0, 0.0), // +x
        make_color(0.8, 0.0, 0.0), // -x
        make_color(1.0, 0.2, 0.2), // +y
        make_color(0.6, 0.0, 0.0), // -y
        make_color(0.9, 0.0, 0.0), // +z
        make_color(0.7, 0.0, 0.0), // -z
    ];
    // ID 2: 초록 계열 (바닥/천장)
    palette[2] = [
        make_color(0.0, 1.0, 0.0), make_color(0.0, 0.8, 0.0),
        make_color(0.2, 1.0, 0.2), make_color(0.0, 0.6, 0.0),
        make_color(0.0, 0.9, 0.0), make_color(0.0, 0.7, 0.0),
    ];
    // ID 3: 파랑 계열 (테두리)
    palette[3] = [
        make_color(0.0, 0.0, 1.0), make_color(0.0, 0.0, 0.8),
        make_color(0.2, 0.2, 1.0), make_color(0.0, 0.0, 0.6),
        make_color(0.0, 0.0, 0.9), make_color(0.0, 0.0, 0.7),
    ];
    // ID 4: 노랑 (핵)
    palette[4] = [make_color(1.0, 1.0, 0.0); 6]; // 모든 면이 샛노랑
    
    // ID 5: 회색 (선택된 블록 하이라이트용)
    palette[5] = [make_color(0.6, 0.6, 0.6); 6]; // 모든 면이 회색

    (world, palette)
}

// --- 핵심: 메싱 알고리즘 (Face Culling) ---
#[allow(dead_code)]
fn generate_mesh(
    world: &[[[u8; GRID_SIZE]; GRID_SIZE]; GRID_SIZE],
    palette: &[[[f32; 3]; 6]; 6]
) -> Vec<f32> {
    generate_mesh_with_selection(world, palette, None)
}

// 선택된 블록을 고려한 메싱 알고리즘
fn generate_mesh_with_selection(
    world: &[[[u8; GRID_SIZE]; GRID_SIZE]; GRID_SIZE],
    palette: &[[[f32; 3]; 6]; 6],
    selected: Option<(usize, usize, usize)>,
) -> Vec<f32> {
    let mut vertices = Vec::new();

    for x in 0..GRID_SIZE {
        for y in 0..GRID_SIZE {
            for z in 0..GRID_SIZE {
                let block_id = world[x][y][z];
                if block_id == 0 { continue; } // 공기는 그리지 않음
                
                // 선택된 블록이면 팔레트 5번 사용, 아니면 원래 팔레트 사용
                let palette_id = if selected == Some((x, y, z)) {
                    5usize
                } else {
                    block_id as usize
                };

                // 6면 검사 (Face Culling)
                // 이웃이 그리드 밖이거나(경계면), 이웃이 공기(0)일 때만 그림
                
                // +x (Right)
                if x + 1 >= GRID_SIZE || world[x+1][y][z] == 0 {
                    add_face(&mut vertices, x, y, z, FACE_RIGHT, palette[palette_id][FACE_RIGHT]);
                }
                // -x (Left)
                if x == 0 || world[x-1][y][z] == 0 {
                    add_face(&mut vertices, x, y, z, FACE_LEFT, palette[palette_id][FACE_LEFT]);
                }
                // +y (Top)
                if y + 1 >= GRID_SIZE || world[x][y+1][z] == 0 {
                    add_face(&mut vertices, x, y, z, FACE_TOP, palette[palette_id][FACE_TOP]);
                }
                // -y (Bottom)
                if y == 0 || world[x][y-1][z] == 0 {
                    add_face(&mut vertices, x, y, z, FACE_BOTTOM, palette[palette_id][FACE_BOTTOM]);
                }
                // +z (Front)
                if z + 1 >= GRID_SIZE || world[x][y][z+1] == 0 {
                    add_face(&mut vertices, x, y, z, FACE_FRONT, palette[palette_id][FACE_FRONT]);
                }
                // -z (Back)
                if z == 0 || world[x][y][z-1] == 0 {
                    add_face(&mut vertices, x, y, z, FACE_BACK, palette[palette_id][FACE_BACK]);
                }
            }
        }
    }
    vertices
}

// 단일 면(사각형 = 삼각형 2개)을 정점 리스트에 추가
fn add_face(vertices: &mut Vec<f32>, ix: usize, iy: usize, iz: usize, face_idx: usize, color: [f32; 3]) {
    // 논리 좌표를 월드 절대 좌표로 변환
    // 시작점(-0.5) + 인덱스 * 크기(0.2)
    let x = WORLD_ORIGIN + (ix as f32) * BLOCK_SIZE;
    let y = WORLD_ORIGIN + (iy as f32) * BLOCK_SIZE;
    let z = WORLD_ORIGIN + (iz as f32) * BLOCK_SIZE;
    let s = BLOCK_SIZE; // size shortcut

    // 각 면의 로컬 좌표 정의 (Triangle 1 + Triangle 2)
    let quad_pos: [f32; 18] = match face_idx {
        FACE_RIGHT => [ // +x face
            x+s, y,   z,
            x+s, y+s, z,
            x+s, y+s, z+s,
            x+s, y+s, z+s,
            x+s, y,   z+s,
            x+s, y,   z,
        ],
        FACE_LEFT => [ // -x face
            x, y,   z+s,
            x, y+s, z+s,
            x, y+s, z,
            x, y+s, z,
            x, y,   z,
            x, y,   z+s,
        ],
        FACE_TOP => [ // +y face
            x,   y+s, z+s,
            x+s, y+s, z+s,
            x+s, y+s, z,
            x+s, y+s, z,
            x,   y+s, z,
            x,   y+s, z+s,
        ],
        FACE_BOTTOM => [ // -y face
            x,   y, z,
            x+s, y, z,
            x+s, y, z+s,
            x+s, y, z+s,
            x,   y, z+s,
            x,   y, z,
        ],
        FACE_FRONT => [ // +z face
            x,   y,   z+s,
            x+s, y,   z+s,
            x+s, y+s, z+s,
            x+s, y+s, z+s,
            x,   y+s, z+s,
            x,   y,   z+s,
        ],
        FACE_BACK => [ // -z face
            x+s, y,   z,
            x,   y,   z,
            x,   y+s, z,
            x,   y+s, z,
            x+s, y+s, z,
            x+s, y,   z,
        ],
        _ => [0.0; 18],
    };

    // 정점 6개에 대해 (위치 3개 + 색상 3개) 씩 push
    for i in 0..6 {
        vertices.push(quad_pos[i*3]);     // x
        vertices.push(quad_pos[i*3 + 1]); // y
        vertices.push(quad_pos[i*3 + 2]); // z
        vertices.push(color[0]);          // r
        vertices.push(color[1]);          // g
        vertices.push(color[2]);          // b
    }
}

// --- 레이캐스팅: 카메라가 바라보는 블록 찾기 ---
fn raycast_block(
    camera: &Camera,
    world: &[[[u8; GRID_SIZE]; GRID_SIZE]; GRID_SIZE],
    max_distance: f32,
) -> Option<(usize, usize, usize)> {
    let ray_origin = camera.position;
    let ray_dir = camera.get_forward_direction();
    
    // DDA (Digital Differential Analyzer) 알고리즘 사용
    let step_size = 0.02; // 레이 진행 스텝 크기
    let mut t = 0.0;
    
    while t < max_distance {
        let point = ray_origin + ray_dir * t;
        
        // 월드 좌표를 그리드 인덱스로 변환
        let ix = ((point.x - WORLD_ORIGIN) / BLOCK_SIZE).floor() as i32;
        let iy = ((point.y - WORLD_ORIGIN) / BLOCK_SIZE).floor() as i32;
        let iz = ((point.z - WORLD_ORIGIN) / BLOCK_SIZE).floor() as i32;
        
        // 그리드 범위 내인지 확인
        if ix >= 0 && ix < GRID_SIZE as i32 &&
           iy >= 0 && iy < GRID_SIZE as i32 &&
           iz >= 0 && iz < GRID_SIZE as i32 {
            let block_id = world[ix as usize][iy as usize][iz as usize];
            if block_id != 0 {
                // 공기가 아닌 블록 발견
                return Some((ix as usize, iy as usize, iz as usize));
            }
        }
        
        t += step_size;
    }
    
    None // 블록을 찾지 못함
}

// --- GL Helper Functions ---
unsafe fn create_shader_program() -> u32 {
    let vertex_shader = unsafe { compile_shader(VERTEX_SHADER_SRC, gl::VERTEX_SHADER) };
    let fragment_shader = unsafe { compile_shader(FRAGMENT_SHADER_SRC, gl::FRAGMENT_SHADER) };
    
    let program = unsafe { gl::CreateProgram() };
    unsafe { gl::AttachShader(program, vertex_shader) };
    unsafe { gl::AttachShader(program, fragment_shader) };
    unsafe { gl::LinkProgram(program) };
    
    unsafe { gl::DeleteShader(vertex_shader) };
    unsafe { gl::DeleteShader(fragment_shader) };
    
    program
}

unsafe fn compile_shader(src: &str, shader_type: u32) -> u32 {
    let shader = unsafe { gl::CreateShader(shader_type) };
    let c_str = CString::new(src.as_bytes()).unwrap();
    unsafe { gl::ShaderSource(shader, 1, &c_str.as_ptr(), ptr::null()) };
    unsafe { gl::CompileShader(shader) };
    shader
}

unsafe fn set_mat4(program: u32, name: &str, mat: &Mat4) {
    let c_name = CString::new(name).unwrap();
    let loc = unsafe { gl::GetUniformLocation(program, c_name.as_ptr()) };
    unsafe { gl::UniformMatrix4fv(loc, 1, gl::FALSE, mat.to_cols_array().as_ptr()) };
}

unsafe fn create_crosshair_shader_program() -> u32 {
    let vertex_shader = unsafe { compile_shader(CROSSHAIR_VERTEX_SHADER_SRC, gl::VERTEX_SHADER) };
    let fragment_shader = unsafe { compile_shader(CROSSHAIR_FRAGMENT_SHADER_SRC, gl::FRAGMENT_SHADER) };
    
    let program = unsafe { gl::CreateProgram() };
    unsafe { gl::AttachShader(program, vertex_shader) };
    unsafe { gl::AttachShader(program, fragment_shader) };
    unsafe { gl::LinkProgram(program) };
    
    unsafe { gl::DeleteShader(vertex_shader) };
    unsafe { gl::DeleteShader(fragment_shader) };
    
    program
}