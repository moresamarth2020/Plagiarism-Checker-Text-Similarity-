from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def plagiarism_checker():
    print("----- PLAGIARISM CHECKER -----\n")

    text1 = input("Enter first text: ")
    text2 = input("Enter second text: ")

    texts = [text1, text2]

    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform(texts)

    similarity = cosine_similarity(vectors)[0][1] * 100

    print("\n----- RESULT -----")
    print(f"Similarity: {similarity:.2f}%")

    if similarity > 70:
        print("⚠ High chance of plagiarism")
    elif similarity > 40:
        print("⚠ Moderate similarity")
    else:
        print("✔ Low similarity")

plagiarism_checker()
