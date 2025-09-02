# File in which we store the knowledge base in terms of prompt tuning
# for our CLIP model. This file will be used to generate labels for images.


# We instantiate a simple dictionary to store the prompts for each concept. 
# For this task, only fruits are considered, and each concept is a type of fruit 
# disease or disorder. You can add more concepts and prompts to this dictionary to 
# expand the knowledge base to your needs.
captions = {

    "fruits": [
        # fungal_diseases
        "a fruit with green mold",  
        "a fruit with blue mold",
        "a fruit with gray mold",
        "a fruit with black mold",
        "a fruit with white mold",

        # bacterial_diseases
        "a fruit with bacterial spots",
        "a fruit with bacterial infection",
        "a fruit with soft rot",

        # viral_diseases
        "a fruit with viral mosaic",
        "a fruit with viral rings",
        "a fruit with viral discoloration",

        # physical_damage
        "a fruit with bruises",
        "a fruit with surface scars",
        "a fruit with abrasions",
        "a fruit with deformations"
        "a fruit with storage damage",
        "a fruit with missing parts"

        # aging_symptoms
        "a fruit with overripening",
        "a fruit with wrinkled skin",
        
        # texture
        "a fruit with soft spots",
        "a fruit with firm texture",
        "a fruit with rough surface",
        "a fruit with smooth texture",
        "a fruit with shriveled skin",

        # aging
        "an old fruit",
        "an aged fruit",
        "a fruit with age spots",
        "a fruit with fading color",
        "a fruit with dry, wrinkled skin"
    ]
}




