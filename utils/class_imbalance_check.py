import os

def count_images_in_classes(root_dir, extensions=['.jpg', '.png', '.gif']):
    class_count = {}

    for class_dir in os.listdir(root_dir):
        full_class_path = os.path.join(root_dir, class_dir)
        
        if os.path.isdir(full_class_path):
            image_count = 0
            for file_name in os.listdir(full_class_path):
                if any(file_name.endswith(ext) for ext in extensions):
                    image_count += 1
            
            class_count[class_dir] = image_count

    min_class = min(class_count, key=class_count.get)
    max_class = max(class_count, key=class_count.get)
    avg_count = sum(class_count.values()) / len(class_count)
    
    sorted_counts = dict(sorted(class_count.items(), key=lambda item: item[1]))
    least_10_classes = dict(list(sorted_counts.items())[:10])

    return len(class_count), sum(class_count.values()), avg_count, min_class, class_count[min_class], max_class, class_count[max_class], least_10_classes


if __name__ == "__main__":
    name = 'orig'
    root_directory = "/path/to/datasets/orig/CUB_200_2011/images"
    num_classes, total, avg, min_class, min_count, max_class, max_count, least_10 = count_images_in_classes(root_directory)

    print(f"Total classes: {num_classes}")
    print(f"Total images: {total}")
    print(f"Average images per class: {avg}")
    print(f"Class with maximum images: {max_class} ({max_count} images)")
    print(f"Class with minimum images: {min_class} ({min_count} images)")
    print(f"10 classes with least images: {least_10}")