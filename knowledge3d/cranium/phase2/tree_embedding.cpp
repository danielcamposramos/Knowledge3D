#include <vector>
#include <random>
#include <algorithm>
#include <cmath>

struct TreeNode {
    std::vector<float> position;     // xyz
    std::vector<float> embedding;    // concept embedding
    std::vector<TreeNode> children;
    float honesty;                   // RLWHF honesty score
};

class FractalTreeGenerator {
public:
    static TreeNode generateTreeFromEmbedding(const std::vector<float>& root_emb) {
        TreeNode root;
        root.position = {0.0f, 0.0f, 0.0f};
        root.embedding = root_emb;
        root.honesty = (root_emb.size() > 72) ? root_emb[72] : 1.0f;

        int max_depth = 3;
        if (root_emb.size() > 3) {
            max_depth = std::max(1, std::min(5, (int)std::floor(root_emb[2] * 4.0f + 1.0f)));
        }
        generateChildren(root, 1, max_depth, root_emb);
        return root;
    }

private:
    static void generateChildren(TreeNode& parent, int current_depth, int max_depth, const std::vector<float>& parent_emb) {
        if (current_depth >= max_depth) return;
        for (int i = 0; i < 3; ++i) {
            TreeNode child;
            child.embedding = mutateEmbedding(parent_emb, current_depth, i);
            child.honesty = (child.embedding.size() > 72) ? child.embedding[72] : 1.0f;
            float angle = (float)(i - 1) * 0.6f; // -0.6, 0, +0.6
            float len = 1.0f;
            if (parent_emb.size() > 1) {
                len = std::max(0.5f, std::min(2.0f, parent_emb[1] * 1.5f + 0.5f));
            }
            child.position = {
                parent.position[0] + std::sin(angle) * len,
                parent.position[1] + std::cos(angle) * len,
                parent.position[2] + 0.5f
            };
            generateChildren(child, current_depth + 1, max_depth, child.embedding);
            parent.children.push_back(std::move(child));
        }
    }

    static std::vector<float> mutateEmbedding(const std::vector<float>& parent_emb, int depth, int seed) {
        std::vector<float> out = parent_emb;
        std::mt19937 rng((unsigned int)(depth * 1337u + (unsigned int)seed));
        std::normal_distribution<float> noise(0.0f, 0.05f * std::max(1, depth));
        for (auto &v : out) v = v + noise(rng);
        return out;
    }
};

