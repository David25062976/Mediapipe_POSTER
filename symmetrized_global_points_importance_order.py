import torch

def symmetrize_importance(input_pt_path, output_pt_path):
    # 1. 讀取原本的排序表
    print(f"載入原始排序表: {input_pt_path}")
    original_order = torch.load(input_pt_path, weights_only=True)
    print(original_order)
    num_points = len(original_order)
    
    # 2. 將「名次」轉換為「分數」(名次越前面，分數越高)
    # rank 0 (第一名) -> score 478
    scores = torch.zeros(num_points)
    for rank, pt_idx in enumerate(original_order):
        scores[pt_idx.item()] = num_points - rank

    # 3. 貼上我們之前整理好的左右臉對稱表
    symmetry_map = {
        246: 466, 161: 388, 160: 387, 159: 386, 158: 385, 157: 384, 173: 398, 33: 263, 
        7: 249, 163: 390, 144: 373, 145: 374, 153: 380, 154: 381, 155: 382, 133: 362, 
        468: 473, 469: 474, 470: 475, 471: 476, 472: 477, 70: 300, 63: 293, 105: 334, 
        66: 296, 107: 336, 55: 285, 65: 295, 52: 282, 53: 283, 46: 276, 102: 331, 
        49: 279, 48: 278, 115: 344, 129: 358, 198: 429, 217: 437, 209: 420, 131: 360, 
        326: 97, 61: 291, 185: 409, 40: 270, 39: 269, 37: 267, 78: 308, 191: 415, 
        80: 310, 81: 311, 82: 312, 146: 375, 91: 321, 181: 405, 84: 314, 95: 324, 
        88: 318, 178: 402, 87: 317, 234: 454, 93: 323, 132: 361, 58: 288, 172: 397, 
        136: 365, 150: 379, 149: 378, 176: 400, 148: 377, 21: 251, 54: 284, 67: 297, 
        68: 298, 69: 299, 71: 301, 103: 332, 104: 333, 108: 337, 109: 338, 36: 266, 
        205: 425, 206: 426, 207: 427, 187: 411, 123: 352, 116: 345, 117: 346, 118: 347, 
        119: 348, 100: 329, 47: 277, 127: 356, 162: 389
    }

    # 4. 強制對稱：取左右兩點的最高分
    sym_scores = scores.clone()
    for pt_left, pt_right in symmetry_map.items():
        max_score = max(scores[pt_left].item(), scores[pt_right].item())
        sym_scores[pt_left] = max_score
        sym_scores[pt_right] = max_score

    # 5. 根據新的對稱分數重新排序 (由高到低)
    # sorted_indices 就是我們新的 global_points_importance_order
    _, sorted_indices = torch.sort(sym_scores, descending=True)

    # 6. 儲存新檔案
    torch.save(sorted_indices, output_pt_path)
    print(f"✅ 對稱化完成！新排序已儲存至: {output_pt_path}")

    # 印出前 10 個點檢查是否成雙成對
    print("檢查前 20 名的重要點 (應該會看到成對出現):")
    print(sorted_indices[:20].tolist())

if __name__ == "__main__":
    # 將路徑替換成你實際的路徑
    INPUT_PATH = "./checkpoint/20260604-014225_phase1/global_points_importance_order.pt"
    OUTPUT_PATH = "./checkpoint/20260604-014225_phase1/symmetrized_global_points_importance_order.pt"
    
    symmetrize_importance(INPUT_PATH, OUTPUT_PATH)