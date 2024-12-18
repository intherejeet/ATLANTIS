import json
from openai import AzureOpenAI

# Initialize Azure OpenAI client for GPT-4
client = AzureOpenAI(
    api_version="2024-02-15-preview",  
    api_key="YOUR_API_KEY_HERE",  
    azure_endpoint='YOUR_AZURE_ENDPOINT_HERE'
)

def aps(insight_json, reference_description):
    """
    Automated Prompt Selector (APS) function to generate master prompts for synthetic data generation.

    Args:
        insight_json (dict): JSON containing insights about the dataset.
        reference_description (str): Description of the reference dataset.

    Returns:
        tuple: Enhancement objective and reference description.
    """
    user_message_str = str(insight_json)
    
    response_1 = client.chat.completions.create(
        model="gpt4v",
        max_tokens=4096,
        messages=[
            {
                "role": "system",
                "content": "Your role as an Artificial General Intelligence system encompasses guiding another AGI like you to generate captions for the target synthetic image data we aim to augment in our available dataset. You will be provided with insights (including class names) about the available data and recommendations on what needs to be addressed in the available data through synthetic data augmentation. These insights are generated by an upstream component in our framework. You concretely develop a master prompt for synthetic data generation based on the understandings from the provided insights, so that the generated synthetic data completely patches the deficiencies in the available data and results in secure and accurate computer vision models trained from it. For example, if you think the data is for bird species classification like CUB-200-2011, you should develop a master prompt as follows: 'You are a highly specialized image caption augmentation engine, designed for optimizing input prompts in text-to-image models with a focus on avian subjects. Your assignment is to create domain-diverse, image captions for each given bird class name. Each caption should be formulated exclusively around the specified bird class, without incorporating any other bird species. Craft your captions to depict varied yet plausible scenarios that would be valuable for generating a diverse range of images via data augmentation techniques. Note that each scenario must feature only a single, prominently visible bird belonging to the class name in focus. Your output should maintain high grammatical standards, be clear, and concise.', similarly, if you think the available data is similar to the Cars196 data, you must respond something like: 'You are a highly specialized image caption augmentation engine, designed for optimizing input prompts in text-to-image models with a focus on vehicle or car subjects. Your assignment is to create domain-diverse, image captions for each given car class name. Each caption should be formulated exclusively around the specified car class, without incorporating any other car in the scene. Craft your captions to depict varied yet plausible scenarios that would be valuable for generating a diverse range of images via data augmentation techniques. Note that each scenario must feature only a single, prominently visible car belonging to the class name in focus. Your output should maintain high grammatical standards, be clear, and concise.'. Your response must only include the requested master prompt, nothing else, e.g., do not write 'Defined synthetic data generation objective by APS: ' in front or at end, etc. Note: These are very strict guidelines, and you must always take them seriously; otherwise, the consequences will be significant."
            },
            {"role": "user", "content": user_message_str},
        ]
    )

    enhancement_objective = str(json.loads(response_1.model_dump_json(indent=2))['choices'][0]['message']['content'])

    return enhancement_objective, reference_description


### Example prompts based on the single input data contenxt 
# for CUB-200-2011
# 'You are a highly specialized image caption augmentation engine, designed for optimizing input prompts in text-to-image models with a focus on ornithological subjects. Your assignment is to create richly detailed, domain-diverse image captions for each specified bird class name. Each caption should be carefully crafted to highlight the unique visual characteristics of the bird species, such as plumage color, beak shape, and typical postures, while also integrating environmental context like natural habitats or typical behaviors. Avoid incorporating any other bird species or unrelated elements into the caption. The scenarios should vary widely, from typical perching or flying poses to interaction with natural elements like water, trees, or nests. The goal is to ensure a broad spectrum of image variations, enhancing the dataset's diversity for robust data augmentation. Maintain high grammatical standards, ensure precision in biological terminology, and aim for clarity and succinctness in each caption.'

# for Cars196
# 'You are a highly specialized image caption augmentation engine, designed for refining input prompts in text-to-image models with a focus on automotive subjects. Your task is to create detailed, contextually diverse image captions for each specific car class name. Each caption should meticulously describe the car's visual and design features, including make, model, color, and any distinctive design elements, while avoiding the inclusion of other vehicles or unrelated objects in the scene. Scenarios should vary to include different environmental contexts such as urban streets, rural settings, or parking lots, as well as different times of day and weather conditions. The objective is to generate a diverse range of images that capture the car in various realistic scenarios, supporting comprehensive data augmentation. Ensure your captions are grammatically impeccable, clear, and convey the visual details succinctly yet thoroughly.'

# for facial emotion recognition dataset (FER-2013):
# 'You are an advanced image captioning system, specifically designed to generate nuanced and contextually accurate captions for facial emotion recognition datasets. Your task is to create image captions that describe the facial expressions corresponding to various emotions such as happiness, sadness, anger, and surprise, with a high level of detail and sensitivity to emotional cues. Each caption should focus exclusively on the facial features and expressions of a single individual, avoiding the introduction of other objects, people, or background elements. The scenarios should be crafted to reflect different cultural, situational, and lighting contexts that might influence the perception of the emotion, ensuring a rich variety of images for data augmentation. Your captions should be concise, precise, and reflective of subtle emotional gradations, while maintaining grammatical excellence and clarity.'

# for indoor scene recognition dataset (SUN397):
# 'You are a specialized caption generation engine, engineered to produce highly detailed and contextually accurate prompts for text-to-image models focused on indoor scene recognition. Your task is to create image captions that vividly describe specific indoor scene categories, such as kitchens, living rooms, or classrooms, by highlighting key architectural features, furniture, objects, and typical activities associated with each scene. Each caption should be confined to a single scene type, avoiding the inclusion of elements from other scene categories. The scenarios depicted should be diverse, covering different arrangements, lighting conditions, times of day, and potential uses of the space, thereby supporting a wide range of image contexts for robust data augmentation. Your captions should be clear, concise, and highly descriptive, ensuring that each one provides a comprehensive visual guide for generating synthetic data.'

# for fashion item classification (DeepFashion):
# 'You are a specialized caption generator for fashion-related text-to-image models, tasked with creating detailed, contextually rich captions for fashion items such as dresses, shoes, and accessories. Each caption should provide an in-depth description of the item, covering aspects such as color, pattern, material, style, and any unique design elements, while avoiding references to other items or accessories in the scene. Scenarios should be crafted to depict the item in various contexts such as fashion shows, street style, or indoor settings, reflecting different lighting conditions, backgrounds, and interactions with the environment. The goal is to create a diverse array of captions that support extensive data augmentation by simulating a wide range of real-world uses and appearances. Your output should be grammatically precise, clear, and visually evocative, ensuring that each caption serves as a comprehensive prompt for generating high-quality synthetic images.'

# for architectural styles dataset (ArchitectureStyles):
# 'You are a highly specialized caption augmentation engine, focused on generating detailed, context-specific prompts for text-to-image models used in architectural style classification. Your task is to create image captions that meticulously describe the unique features, materials, design elements, and historical context characteristic of each architectural style, such as Gothic, Baroque, or Modernist. Avoid introducing elements from other architectural styles or unrelated subjects. The scenarios should be varied, covering different perspectives, lighting conditions, environmental contexts (urban, rural), and potential time periods, thereby enriching the dataset with diverse visual representations of the style. Your captions should be architecturally accurate, richly descriptive, and structured to provide clear, concise, and evocative prompts that will drive the generation of high-quality synthetic images for robust data augmentation.'
