import os

def label_order(root_dir, output_txt_path):
    image_extensions = ('.jpg', '.jpeg', '.png')
    label_entries = []

    for label in range(8):
        label_path = os.path.join(root_dir, str(label))
        if not os.path.isdir(label_path):
            continue

        for filename in os.listdir(label_path):
            if filename.lower().endswith(image_extensions):
                label_entries.append((filename, label))

    # 排序：先依 label，再依圖片檔名數字順序排序
    label_entries.sort(key=lambda x: (x[1], int(x[0].split('.')[0]) if x[0].split('.')[0].isdigit() else x[0]))

    with open(output_txt_path, 'w') as f:
        for fname, label in label_entries:
            f.write(f"{fname} {label}\n")

    print(f"✅ 已寫入 {len(label_entries)} 筆資料到 {output_txt_path}")

# 🧪 範例用法
label_order(
    '/home/lab702/POSTER_V2-main/data/AffectNet/train',
    '/home/lab702/POSTER_V2-main/output_labels/train_annotations_8class.txt'
)
