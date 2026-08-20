from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")
ref = "Photosynthesis converts sunlight into chemical energy."
student = "Plants make energy from sunlight through photosynthesis."

emb = model.encode([student, ref], convert_to_tensor=True)
score = util.cos_sim(emb[0], emb[1]).item()
print(f"Similarity score: {score:.4f}")
