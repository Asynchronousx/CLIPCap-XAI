# This file will be used to generate labels for images.
# You can add your own concepts and prompts to expand the knowledge base to your needs.
# To the scope of this task, only fruits are considered, and each concept is a type 
# of fruit disease or disorder.


# Function that given a list of pairs (caption, probability) returns a synthesized caption
# built in an hierarchical way, from the most severe to the least severe condition.
def synthesize(captions_with_probabilities):

    # The severity map defines the hierarchy of conditions from most severe to least severe.
    # Each condition is based on the probability threshold required to be classified as such.
    severity_map = [
        ("Very Severe", 0.8),
        ("Severe", 0.5),
        ("Moderate", 0.3),
        ("Possible", 0.1),
        ("Minor", 0.0)
    ]
    
    # Initialize a dictionary to store the conditions for each severity level.
    # This will create a categorized list of conditions based on their severity, 
    # which will range from very severe to minor.
    categorized_conditions = {level: [] for level, _ in severity_map}

    # Iterate over the captions and their probabilities to categorize them based on their severity.
    # So, for each pair (caption, prob)
    for caption, prob in captions_with_probabilities:

        # Since our knowledge is in a pretty standardized format we know which part 
        # to strip from the caption to get the actual condition.
        cleaned_caption = caption.lower().replace("a fruit with", "").strip()

        # For each severity level, check if the probability is higher than the threshold.
        for level, threshold in severity_map:

            # If so, append the condition to the list of conditions for that severity level
            # and break the loop to avoid appending the same condition to multiple levels.
            if prob > threshold:
                categorized_conditions[level].append(cleaned_caption)
                break

    # Initialize a list to store the caption parts that will be used to build the final caption.
    caption_parts = []

    # For each severity level, if there are conditions, build the caption part for that level.
    for severity, conditions in categorized_conditions.items():

        # If there are conditions for this severity level, build the caption part.
        if conditions:

            # Join the conditions with "and" to create a more natural language caption.
            severity_part = " and ".join(conditions)

            # If the severity is "Very Severe", append "with" before the conditions.
            # Otherwise, append the severity level before the conditions.
            if severity == "Very Severe":
                caption_parts.append(f"with {severity_part}")
            else:
                caption_parts.append(f"{severity.lower()} {severity_part}")

    # Return the built caption by joining all the caption parts starting with "A fruit with".
    return "A fruit with " + " ".join(caption_parts)