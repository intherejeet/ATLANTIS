import os
import matplotlib.pyplot as plt

def count_images_in_classes(root_dir, extensions=['.jpg', '.png', '.gif']):
    class_count = {}
    total_images = 0
    min_count = float('inf')
    min_class = None
    images_in_first_100_classes = 0
    images_in_remaining_100_classes = 0
    
    for i in range(1, 201):
        folder_name = f"{i:03d}"
        class_count[folder_name] = 0

    for class_dir in os.listdir(root_dir):
        full_class_path = os.path.join(root_dir, class_dir)
        
        if os.path.isdir(full_class_path):
            image_count = 0
            for file_name in os.listdir(full_class_path):
                if any(file_name.endswith(ext) for ext in extensions):
                    image_count += 1
            
            class_count[class_dir] = image_count
            total_images += image_count
            
            class_index = int(class_dir.split('.')[0])
            if class_index <= 100:
                images_in_first_100_classes += image_count
            else:
                images_in_remaining_100_classes += image_count
            
            if image_count < min_count:
                min_count = image_count
                min_class = class_dir

    sorted_class_count = dict(sorted(class_count.items(), key=lambda x: x[0]))
    
    return sorted_class_count, total_images, min_class, min_count, images_in_first_100_classes, images_in_remaining_100_classes

def plot_class_distribution(class_counts, name):
    plt.figure(figsize=(20, 30))
    plt.barh(list(class_counts.keys()), list(class_counts.values()), color='skyblue')
    plt.xlabel('Number of Samples')
    plt.ylabel('Class Names')
    plt.title('Class-wise Sample Distribution')

    class_names = [key if val > 0 else "" for key, val in class_counts.items()]
    plt.yticks(ticks=range(len(class_names)), labels=class_names, fontsize=6, rotation=0)
    plt.tick_params(axis='y', labelsize=8)
    plt.tight_layout()
    plt.savefig(f"./visualization/class_distribution_{name}.png")
    plt.show()

if __name__ == "__main__":
    num_imgs_syn_list = [None]
    num_imgs_orig_list = [0]
    delta_list = [1, 1.5]
    for num_imgs_src1 in num_imgs_syn_list:
        for num_imgs_src2 in num_imgs_orig_list:
            for delta in delta_list:
                syn_name = 'all' if num_imgs_src1 is None else num_imgs_src1
                orig_name = 'all' if num_imgs_src2 is None else num_imgs_src2

                name = f'syn_{syn_name}_orig_{orig_name}_delta_{delta}'
                
                root_directory = f"path/to/merged/{name}/images"  # Replace with your actual path
                
                counts, total, min_class, min_count, first_100, remaining_100 = count_images_in_classes(root_directory)

                print(f"Total images: {total}")
                print(f"Images in first 100 classes: {first_100}")
                print(f"Images in remaining 100 classes: {remaining_100}")
                print(f"Class with minimum images: {min_class} ({min_count} images)")

                plot_class_distribution(counts, name)
