#version 450 core
// Leaf glow based on concept similarity (cyan pulse)
in vec3 world_position;
in vec3 embedding_normal;
in vec4 embedding_color;
in float embedding_roughness;
in float embedding_metallic;
in float concept_similarity; // 0..1
uniform vec3 light_direction;
uniform vec3 camera_position;
out vec4 final_color;

void main() {
    vec3 normal = normalize(embedding_normal);
    vec3 view_dir = normalize(camera_position - world_position);
    vec3 light_dir = normalize(-light_direction);
    float ndotl = max(dot(normal, light_dir), 0.0);
    vec3 diffuse = embedding_color.rgb * ndotl;
    vec3 reflect_dir = reflect(-light_dir, normal);
    float spec_power = pow(max(1.0 - embedding_roughness, 0.0), 2.0) * 256.0;
    float spec = pow(max(dot(view_dir, reflect_dir), 0.0), spec_power);
    vec3 specular = mix(vec3(0.04), embedding_color.rgb, clamp(embedding_metallic, 0.0, 1.0)) * spec;
    vec3 result = diffuse + specular;
    if (concept_similarity > 0.0) {
        float glow = concept_similarity * 2.0;
        result += vec3(0.0, 1.0, 1.0) * glow;
    }
    final_color = vec4(result, embedding_color.a);
}

