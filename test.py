import cyberbully

# Quick check
is_bully = cyberbully.is_cyberbullying("You are a wonderful person!")
print(is_bully)  # False

# Detailed analysis
res = cyberbully.predict("Nobody likes you, you are ugly and pathetic")
print(res.label)           # 'cyberbullying'
print(res.score)           # 0.9997
print(res.category)        # 'targeted_insult_general'
print(res.category_scores) # top 5 category probabilities
