import streamlit as st
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer, util
import torch
import time

# Load the SentenceTransformer model
model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

def transcribe_audio(video_path):
    start_time = time.time()
    # Transcribe the audio from the video
    model = WhisperModel("small")
    segments, info = model.transcribe(video_path, task="translate", beam_size=5, best_of=5)

    # Concatenate the transcription segments into a single text
    text = ''
    for segment in segments:
        text += segment.text

    end_time = time.time()
    execution_time = end_time - start_time
    st.write("execution time-predict category")
    st.write(execution_time)
    return text

def entity_recognition(text, embeddings_categories):
    # Compute embeddings for the text
    start_time = time.time()
    embeddings_text = model.encode([text])
    
    # Compute cosine similarity
    similarities = util.cos_sim(embeddings_text, embeddings_categories)
    
    # Get the index of the maximum similarity
    max_index = torch.argmax(similarities)
    
    # Define the list of categories
    categories = [
        # Technology and IT
    "Software Development", "Web Development", "Mobile App Development",
    "Database Management", "Network Security", "Cloud Computing",
    "Artificial Intelligence", "Machine Learning", "Data Science",
    "Cybersecurity", "Information Systems", "IT Management",
    "DevOps", "Blockchain Technology",

    # Business and Management
    "Business Administration", "Management", "Entrepreneurship",
    "Marketing Strategy", "Sales Management", "Business Analysis",
    "Finance Management", "Accounting", "Economics", "Human Resource Management",
    "Supply Chain Management", "Strategic Management",

    # Design and Creative Arts
    "Graphic Design", "UI/UX Design", "Animation",
    "Illustration", "Digital Painting", "Photography",
    "Video Editing", "Music Production", "Interior Design",
    "Fashion Design", "Sculpture", "Printmaking", "Ceramics",
    "Drawing", "Calligraphy", "Mixed Media",

    # Education and Teaching
    "Curriculum Development", "Educational Psychology", "Instructional Design",
    "Classroom Management", "Pedagogy", "Special Education",
    "Language Teaching", "STEM Education",

    # Social Sciences
    "Sociology", "Psychology", "Anthropology",
    "Political Science", "International Relations",
    "Criminology", "Geography", "Demography",

    # Health and Wellness
    "Nutrition", "Fitness Training", "Yoga",
    "Mental Health Counseling", "Physical Therapy",
    "Dietetics", "Public Health",

    # Science and Engineering
    "Physics", "Chemistry", "Biology",
    "Environmental Science", "Civil Engineering",
    "Electrical Engineering", "Mechanical Engineering",
    "Aerospace Engineering",

    # Language Learning
    "English Language", "Spanish Language", "French Language",
    "Chinese Language", "Japanese Language", "German Language",
    "Italian Language", "Arabic Language", "Russian Language",
    "Korean Language",

    # Arts and Humanities
    "Literature", "History", "Philosophy",
    "Religious Studies", "Fine Arts", "Performing Arts",
    "Archaeology", "Linguistics", "Cultural Studies",

    # Mathematics and Statistics
    "Algebra", "Calculus", "Geometry",
    "Probability & Statistics", "Number Theory",

    # Finance and Economics
    "Investment Banking", "Financial Analysis",
    "Econometrics", "Corporate Finance",
    "Financial Risk Management",

    # Miscellaneous
    "Project Management", "Public Speaking",
    "Time Management", "Critical Thinking",
    "Problem Solving", "Creativity",
    "Leadership Development", "Digital Marketing",
    "Entrepreneurship", "Personal Finance", "Event Planning",

    # Sports and Recreation
    "Soccer", "Basketball", "Football", "Hockey",
    "Golf", "Tennis", "Swimming", "Cycling",
    "Running", "Gymnastics",

    # Entertainment and Media
    "Film Studies", "Television Production", "Acting",
    "Directing", "Screenwriting", "Music Theory",
    "Sound Design", "Broadcasting", "Journalism",

    # Crafts and DIY
    "Woodworking", "Knitting", "Crocheting", "Jewelry Making",
    "Candle Making", "Soap Making", "Pottery", "Origami",
    "Scrapbooking", "Leatherworking",

    # Culinary Arts
    "Cooking Techniques", "Baking", "Pastry Making",
    "Culinary Nutrition", "Food Photography", "Mixology",
    "Gastronomy", "Food Styling", "Menu Planning",
    "Food Writing",

    # Nature and Environment
    "Environmental Conservation", "Sustainability",
    "Ecology", "Wildlife Photography", "Botany",
    "Zoology", "Marine Biology", "Environmental Policy",

    # Travel and Adventure
    "Travel Photography", "Adventure Sports", "Backpacking",
    "Camping", "Hiking", "Mountaineering", "Wildlife Safaris",
    "Cultural Immersion",

    # Fashion and Beauty
    "Fashion Design", "Makeup Artistry", "Hairstyling",
    "Fashion Styling", "Personal Styling", "Modeling",
    "Cosmetology", "Skincare", "Nail Art",

    # Parenting and Family
    "Parenting Skills", "Child Development",
    "Early Childhood Education", "Parenting Styles",
    "Family Relationships", "Positive Discipline",
    "Homeschooling"
    ]
    
    # Check if max_index is within the valid range
    if 0 <= max_index < len(categories):
        category_name = categories[max_index.item()]
    else:
        category_name = "Unknown"

    end_time = time.time()
    execution_time = end_time - start_time
    st.write("execution time-predict category")
    st.write(execution_time)
    
    return category_name

# Main function to run the Streamlit app
def main():
    # Set up the Streamlit app
    st.title("Video Categorization App")
    st.write("Upload a video file to automatically transcribe and classify its category.")

    # File uploader to upload the video file
    video_file = st.file_uploader("Upload Video File", type=["mp4"])

    if video_file:
        # Transcribe the audio when the video is uploaded
        transcription = transcribe_audio(video_file)

        # Perform entity recognition to classify the category
        category = entity_recognition(transcription, embeddings_categories)

        # Display the transcription
        st.subheader("Transcription:")
        st.write(transcription)

        # Display the predicted category
        st.subheader("Predicted Category:")
        st.success(category)

# Load embeddings for categories
docs = [
    # Technology and IT
    "Software Development", "Web Development", "Mobile App Development",
    "Database Management", "Network Security", "Cloud Computing",
    "Artificial Intelligence", "Machine Learning", "Data Science",
    "Cybersecurity", "Information Systems", "IT Management",
    "DevOps", "Blockchain Technology",

    # Business and Management
    "Business Administration", "Management", "Entrepreneurship",
    "Marketing Strategy", "Sales Management", "Business Analysis",
    "Finance Management", "Accounting", "Economics", "Human Resource Management",
    "Supply Chain Management", "Strategic Management",

    # Design and Creative Arts
    "Graphic Design", "UI/UX Design", "Animation",
    "Illustration", "Digital Painting", "Photography",
    "Video Editing", "Music Production", "Interior Design",
    "Fashion Design", "Sculpture", "Printmaking", "Ceramics",
    "Drawing", "Calligraphy", "Mixed Media",

    # Education and Teaching
    "Curriculum Development", "Educational Psychology", "Instructional Design",
    "Classroom Management", "Pedagogy", "Special Education",
    "Language Teaching", "STEM Education",

    # Social Sciences
    "Sociology", "Psychology", "Anthropology",
    "Political Science", "International Relations",
    "Criminology", "Geography", "Demography",

    # Health and Wellness
    "Nutrition", "Fitness Training", "Yoga",
    "Mental Health Counseling", "Physical Therapy",
    "Dietetics", "Public Health",

    # Science and Engineering
    "Physics", "Chemistry", "Biology",
    "Environmental Science", "Civil Engineering",
    "Electrical Engineering", "Mechanical Engineering",
    "Aerospace Engineering",

    # Language Learning
    "English Language", "Spanish Language", "French Language",
    "Chinese Language", "Japanese Language", "German Language",
    "Italian Language", "Arabic Language", "Russian Language",
    "Korean Language",

    # Arts and Humanities
    "Literature", "History", "Philosophy",
    "Religious Studies", "Fine Arts", "Performing Arts",
    "Archaeology", "Linguistics", "Cultural Studies",

    # Mathematics and Statistics
    "Algebra", "Calculus", "Geometry",
    "Probability & Statistics", "Number Theory",

    # Finance and Economics
    "Investment Banking", "Financial Analysis",
    "Econometrics", "Corporate Finance",
    "Financial Risk Management",

    # Miscellaneous
    "Project Management", "Public Speaking",
    "Time Management", "Critical Thinking",
    "Problem Solving", "Creativity",
    "Leadership Development", "Digital Marketing",
    "Entrepreneurship", "Personal Finance", "Event Planning",

    # Sports and Recreation
    "Soccer", "Basketball", "Football", "Hockey",
    "Golf", "Tennis", "Swimming", "Cycling",
    "Running", "Gymnastics",

    # Entertainment and Media
    "Film Studies", "Television Production", "Acting",
    "Directing", "Screenwriting", "Music Theory",
    "Sound Design", "Broadcasting", "Journalism",

    # Crafts and DIY
    "Woodworking", "Knitting", "Crocheting", "Jewelry Making",
    "Candle Making", "Soap Making", "Pottery", "Origami",
    "Scrapbooking", "Leatherworking",

    # Culinary Arts
    "Cooking Techniques", "Baking", "Pastry Making",
    "Culinary Nutrition", "Food Photography", "Mixology",
    "Gastronomy", "Food Styling", "Menu Planning",
    "Food Writing",

    # Nature and Environment
    "Environmental Conservation", "Sustainability",
    "Ecology", "Wildlife Photography", "Botany",
    "Zoology", "Marine Biology", "Environmental Policy",

    # Travel and Adventure
    "Travel Photography", "Adventure Sports", "Backpacking",
    "Camping", "Hiking", "Mountaineering", "Wildlife Safaris",
    "Cultural Immersion",

    # Fashion and Beauty
    "Fashion Design", "Makeup Artistry", "Hairstyling",
    "Fashion Styling", "Personal Styling", "Modeling",
    "Cosmetology", "Skincare", "Nail Art",

    # Parenting and Family
    "Parenting Skills", "Child Development",
    "Early Childhood Education", "Parenting Styles",
    "Family Relationships", "Positive Discipline",
    "Homeschooling"
]
embeddings_categories = model.encode(docs)

# Run the main function
if __name__ == "__main__":
    main()
