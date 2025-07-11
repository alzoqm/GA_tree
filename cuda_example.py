import torch
import gatree_cuda # 빌드된 모듈 임포트

# ... population, features_per_tree, current_positions 준비 ...

# 1. 입력 텐서 준비
pop_size = population.pop_size
all_features_list = population.population[0].all_features
num_features = len(all_features_list)

features_tensor = torch.zeros(pop_size, num_features, dtype=torch.float32)
for i, features_dict in enumerate(features_per_tree):
    for feat_name, feat_val in features_dict.items():
        feat_idx = all_features_list.index(feat_name)
        features_tensor[i, feat_idx] = feat_val
        
pos_map = {'LONG': 0, 'HOLD': 1, 'SHORT': 2}
positions_tensor = torch.tensor([pos_map[pos] for pos in current_positions], dtype=torch.long)

# 2. 텐서를 GPU로 이동
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
population_tensor_cuda = population.population_tensor.to(device)
features_tensor_cuda = features_tensor.to(device)
positions_tensor_cuda = positions_tensor.to(device)

# =========================================================
# === 변경점: 결과 텐서를 Python에서 미리 생성 ===
results_tensor_cuda = torch.zeros((pop_size, 3), dtype=torch.float32, device=device)
# =========================================================

# 3. CUDA 함수 호출 (이제 반환값이 없음)
gatree_cuda.predict(
    population_tensor_cuda,
    features_tensor_cuda,
    positions_tensor_cuda,
    results_tensor_cuda # 생성한 텐서를 인자로 전달
)

# 4. 결과 확인
# 이제 results_tensor_cuda 에 결과가 채워져 있습니다.
print("CUDA execution complete. Results are now in the pre-allocated tensor:")
print(results_tensor_cuda)
