extern crate glfw;
extern crate gl;

use glfw::{Action, Context, Key};
use glam::{Mat4, Vec3};
use std::mem;
use std::ptr;
use std::ffi::CString;

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

    // 2. GL 로딩
    gl::load_with(|s| {
        window.get_proc_address(s)
            .map(|ptr| ptr as *const _)   // 값이 있으면 포인터로 변환
            .unwrap_or(std::ptr::null())  // 값이 없으면(None) null 포인터 반환
    });

    // 3. 데이터 생성 (월드 데이터 & 팔레트)
    let (world_data, palette) = init_data();

    // 4. 메싱 (Meshing): 데이터를 정점 배열로 변환
    // 위치(3) + 색상(3) = 6 floats per vertex
    let vertices = generate_mesh(&world_data, &palette);
    
    // 생성된 정점이 없다면(전부 공기라면) 종료 방지용 더미 데이터
    if vertices.is_empty() {
        println!("Warning: Mesh is empty!");
    } else {
        println!("Generated Vertices Count: {}", vertices.len() / 6);
        println!("Generated Triangles Count: {}", vertices.len() / 6 / 3);
    }

    let (mut vbo, mut vao) = (0, 0);
    let shader_program;
    
    unsafe {
        gl::Enable(gl::DEPTH_TEST); // 깊이 테스트 켜기 (앞이 뒤를 가리도록)

        shader_program = create_shader_program();

        gl::GenVertexArrays(1, &mut vao);
        gl::GenBuffers(1, &mut vbo);

        gl::BindVertexArray(vao);
        gl::BindBuffer(gl::ARRAY_BUFFER, vbo);
        
        // Vec<f32> 데이터를 GPU로 전송
        gl::BufferData(
            gl::ARRAY_BUFFER,
            (vertices.len() * mem::size_of::<f32>()) as isize,
            vertices.as_ptr() as *const _,
            gl::STATIC_DRAW,
        );

        let stride = (6 * mem::size_of::<f32>()) as i32;
        gl::VertexAttribPointer(0, 3, gl::FLOAT, gl::FALSE, stride, ptr::null()); // Pos
        gl::EnableVertexAttribArray(0);
        gl::VertexAttribPointer(1, 3, gl::FLOAT, gl::FALSE, stride, (3 * mem::size_of::<f32>()) as *const _); // Color
        gl::EnableVertexAttribArray(1);
    }

    // 6. 렌더링 루프
    while !window.should_close() {
        for (_, event) in glfw::flush_messages(&events) {
            if let glfw::WindowEvent::Key(Key::Escape, _, Action::Press, _) = event {
                window.set_should_close(true);
            }
        }

        unsafe {
            gl::ClearColor(0.1, 0.1, 0.1, 1.0);
            gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);

            gl::UseProgram(shader_program);

            let time = glfw.get_time() as f32;

            // Model: Y축 기준 지속 회전
            let model = Mat4::from_rotation_y(time) * Mat4::from_rotation_z(0.3); // X축 살짝 기울여서 입체감 주기
            
            // View: 카메라 약간 뒤로
            let view = Mat4::from_translation(Vec3::new(0.0, 0.0, -2.5));
            
            // Projection
            let projection = Mat4::perspective_rh_gl(45.0f32.to_radians(), 800.0 / 600.0, 0.1, 100.0);

            set_mat4(shader_program, "model", &model);
            set_mat4(shader_program, "view", &view);
            set_mat4(shader_program, "projection", &projection);

            gl::BindVertexArray(vao);
            // 삼각형 개수 = 전체 float 개수 / 6(속성수)
            gl::DrawArrays(gl::TRIANGLES, 0, (vertices.len() / 6) as i32);
        }

        window.swap_buffers();
        glfw.poll_events();
    }
}

// --- 데이터 초기화 함수 ---
fn init_data() -> ([[[u8; GRID_SIZE]; GRID_SIZE]; GRID_SIZE], [[[f32; 3]; 6]; 5]) {
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
    let mut palette = [[[0.0; 3]; 6]; 5];

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

    (world, palette)
}

// --- 핵심: 메싱 알고리즘 (Face Culling) ---
fn generate_mesh(
    world: &[[[u8; GRID_SIZE]; GRID_SIZE]; GRID_SIZE],
    palette: &[[[f32; 3]; 6]; 5]
) -> Vec<f32> {
    let mut vertices = Vec::new();

    for x in 0..GRID_SIZE {
        for y in 0..GRID_SIZE {
            for z in 0..GRID_SIZE {
                let block_id = world[x][y][z];
                if block_id == 0 { continue; } // 공기는 그리지 않음

                // 6면 검사 (Face Culling)
                // 이웃이 그리드 밖이거나(경계면), 이웃이 공기(0)일 때만 그림
                
                // +x (Right)
                if x + 1 >= GRID_SIZE || world[x+1][y][z] == 0 {
                    add_face(&mut vertices, x, y, z, FACE_RIGHT, palette[block_id as usize][FACE_RIGHT]);
                }
                // -x (Left)
                if x == 0 || world[x-1][y][z] == 0 {
                    add_face(&mut vertices, x, y, z, FACE_LEFT, palette[block_id as usize][FACE_LEFT]);
                }
                // +y (Top)
                if y + 1 >= GRID_SIZE || world[x][y+1][z] == 0 {
                    add_face(&mut vertices, x, y, z, FACE_TOP, palette[block_id as usize][FACE_TOP]);
                }
                // -y (Bottom)
                if y == 0 || world[x][y-1][z] == 0 {
                    add_face(&mut vertices, x, y, z, FACE_BOTTOM, palette[block_id as usize][FACE_BOTTOM]);
                }
                // +z (Front)
                if z + 1 >= GRID_SIZE || world[x][y][z+1] == 0 {
                    add_face(&mut vertices, x, y, z, FACE_FRONT, palette[block_id as usize][FACE_FRONT]);
                }
                // -z (Back)
                if z == 0 || world[x][y][z-1] == 0 {
                    add_face(&mut vertices, x, y, z, FACE_BACK, palette[block_id as usize][FACE_BACK]);
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