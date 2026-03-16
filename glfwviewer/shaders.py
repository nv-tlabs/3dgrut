"""GLSL shader sources used by the GLFW viewer screen pass.

The viewer renders the 3DGRUT output into a texture and then presents it
through a minimal fullscreen quad. These constants keep that shader code
centralized and easy to reuse from the window bootstrap module.
"""

VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec2 position;
layout(location = 1) in vec2 texCoord;

out vec2 TexCoord;

void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    TexCoord = texCoord;
}
"""

FRAGMENT_SHADER = """
#version 330 core
in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D texture1;

void main() {
    FragColor = texture(texture1, TexCoord);
}
"""
