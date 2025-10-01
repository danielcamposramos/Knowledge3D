# Embedding Model Comparison

Prompt | Model | Dim | L2 Norm | First 6 Coeffs
---|---|---:|---:|---
energia | qwen3-embedding:4b | 2560 | 1.000 | [-0.000, 0.048, 0.013, -0.009, -0.000, 0.060]
energia | embeddinggemma | 768 | 1.000 | [-0.208, -0.010, 0.025, 0.013, 0.014, 0.028]
energia | snowflake-arctic-embed2 | 1024 | 11.150 | [-0.076, 0.760, 1.058, -0.607, -0.901, -1.088]
conhecimento | qwen3-embedding:4b | 2560 | 1.000 | [0.000, 0.016, 0.003, -0.012, 0.001, 0.039]
conhecimento | embeddinggemma | 768 | 1.000 | [-0.201, -0.011, 0.020, 0.005, 0.010, 0.017]
conhecimento | snowflake-arctic-embed2 | 1024 | 11.570 | [0.429, -0.166, 1.002, -0.541, -0.536, -0.724]
sistemas | qwen3-embedding:4b | 2560 | 1.000 | [0.000, 0.021, -0.022, -0.001, 0.001, 0.043]
sistemas | embeddinggemma | 768 | 1.000 | [-0.198, -0.001, 0.032, 0.001, 0.010, 0.026]
sistemas | snowflake-arctic-embed2 | 1024 | 10.880 | [0.273, -0.206, -0.287, 0.062, 0.056, -0.264]
probability theory | qwen3-embedding:4b | 2560 | 1.000 | [0.000, 0.025, 0.025, 0.001, 0.001, 0.040]
probability theory | embeddinggemma | 768 | 1.000 | [-0.188, -0.010, 0.004, 0.008, -0.009, 0.031]
probability theory | snowflake-arctic-embed2 | 1024 | 12.187 | [-0.531, -0.274, 0.488, 0.529, -0.201, 0.193]
computer vision | qwen3-embedding:4b | 2560 | 1.000 | [0.000, 0.017, 0.001, 0.023, 0.001, 0.028]
computer vision | embeddinggemma | 768 | 1.000 | [-0.200, 0.002, 0.028, 0.006, 0.021, 0.037]
computer vision | snowflake-arctic-embed2 | 1024 | 12.265 | [-0.246, -0.524, 0.280, 0.302, -0.266, 0.702]
