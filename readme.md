# 📊 Multilingual Sentiment Intelligence System  
**Aspect-Based, Explainable & Business-Ready Review Analytics**

---

## 🔍 Project Overview

This project builds an **end-to-end Sentiment Intelligence System** for analyzing customer reviews (English + Bangla).  
Instead of only labeling reviews as positive or negative, the system extracts **deep, actionable insights** such as:

- Overall sentiment
- Aspect-based sentiment (Food, Service, Price, Ambience)
- Rating vs Text conflict detection
- Explainable reasons for dissatisfaction
- Theme-level business insights

The final output is **management-ready**, not just model outputs.

---

## 🎯 Objectives

- Automatically analyze large volumes of customer reviews
- Detect hidden dissatisfaction not visible from ratings alone
- Identify **why** customers are unhappy
- Provide insights usable for **business decision-making**

---

## 🧠 Key Features

### ✅ Multilingual Sentiment Analysis
- Supports **English + Bangla**
- Uses a pretrained Transformer model (`twitter-xlm-roberta-base-sentiment`)

### ✅ Aspect-Based Sentiment (Advanced Core)
Separately analyzes sentiment for:
- 🍽 Food
- 🧑‍💼 Service
- 💰 Price
- 🏠 Ambience

### ✅ Rating ↔ Text Conflict Detection
Identifies cases like:
- ⭐ High rating + negative text → *Hidden Dissatisfaction*
- ⭐ Low rating + positive text → *Politeness Bias*
- ⭐ Neutral rating + strong sentiment → *Ambiguous Experience*

### ✅ Explainable AI (Why is it negative?)
Extracts interpretable reasons such as:
- late
- slow
- rude
- bad food
- cold food  
(including Bangla trigger words)

### ✅ Theme-Level Insights
Aggregates all negative reviews to identify:
- Most frequent complaint themes
- Operational bottlenecks
- Priority areas for improvement

### ✅ Business-Ready Summary
Produces concise insights such as:
- Top negative drivers
- Conflict rate between rating and sentiment
- Actionable improvement signals

---

## 📁 Dataset

- Source: Restaurant customer reviews
- Size: ~1,000 reviews (scalable to 20k+)
- Fields include:
  - `review_text`
  - `review_rating`
  - business metadata (address, rating, etc.)

---

## ⚙️ System Workflow

1. **Load & Clean Data**
   - Remove missing or invalid reviews
   - Normalize text (whitespace, casing)

2. **Overall Sentiment Inference**
   - Predict sentiment + confidence score per review

3. **Aspect Detection**
   - Keyword-based aspect identification
   - Reuse overall sentiment for efficiency

4. **Conflict Detection**
   - Compare numerical rating with textual sentiment

5. **Explainable Reason Extraction**
   - Keyword & tone-based dissatisfaction detection

6. **Theme Aggregation**
   - Group negative reviews by reason

7. **Business Summary Generation**
   - Print management-level insights

---

## 📈 Sample Results

### Overall Sentiment Distribution
- Positive: ~74%
- Negative: ~21%
- Neutral: ~5%

### Rating–Sentiment Conflict Rate
- ~18% of reviews show hidden or ambiguous conflict

### Top Negative Drivers
- Implicit dissatisfaction (tone-based)
- Bad food quality
- Late service
- Slow response
- Cold food

---

## 🛠️ Tech Stack

- **Python**
- **Pandas / NumPy**
- **HuggingFace Transformers**
- **PyTorch**
- **tqdm** (progress tracking)
- **Kaggle Notebook Environment**

---

## 🚀 Why This Matters (Business Impact)

- Goes beyond star ratings
- Detects **silent dissatisfaction**
- Reduces manual review analysis cost
- Enables data-driven service improvements
- Ready for dashboards, reports, or AI assistants

---

## 🔮 Future Extensions

- LLM-powered summarization
- RAG-based conversational analytics
- Trend analysis over time
- Customer churn prediction integration
- Deployment as API or dashboard

---

## 👤 Author

Developed as an **advanced sentiment intelligence system**  
focused on **real-world business applicability**, not just model accuracy.

---

## 📌 Conclusion

This project demonstrates that **sentiment analysis is not just NLP**,  
it is a **decision-support intelligence system** when designed correctly.

