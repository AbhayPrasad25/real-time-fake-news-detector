import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from lime.lime_text import LimeTextExplainer
from newsapi import NewsApiClient

# Load the model and Tokenizer
@st.cache_resource

def load_model():
    model = BertForSequenceClassification.from_pretrained('model\fakenews_model')
    tokenizer = BertTokenizer.from_pretrained('model\fakenews_model')
    return model, tokenizer

model, tokenizer = load_model()

# Load the LIME explainer 
def predict(texts):
    inputs = tokenizer(
        texts,
        padding = True,
        truncation = True,
        max_length = 256,
        return_tensors = 'pt'
    )

    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim = 1)
    return probs.numpy()

explainer = LimeTextExplainer(class_name = ['Fake', 'Real'])

# Making the Stremlit Ui
st.title("Real-Time Fake News Detector 🔍")
st.markdown("Analyze news articles using AI and verify against trusted sources.")

# Get the news article 
news_api_key = st.text_input("Enter NewsAPI Key:", type="password")
category = st.selectbox("Select Category", [
    "business", "entertainment", "general", 
    "health", "science", "sports", "technology"
])

if news_api_key:
    newsapi = NewsApiClient(api_key= news_api_key)

    try:
        headlines = newsapi.get_top_headlines(
            category=category,
            language='en',
            page_size=10
        )
        if not headlines['articles']:
            st.warning("No articles found for the selected category.")
        else:
            st.success(f"Found {len(headlines['articles'])} articles")

            for article in headlines['articles']:
                # skip articles without content
                if not article['content']:
                    continue

                # Display the article
                with st.expander(article['title'][:60] + "..."):
                    st.markdown(f"**Source**: {article['source']['name']}")
                    st.markdown(f"**Published at**: {article['publishedAt']}")

                    # Prediction 
                    content = article['content'] or article['description']
                    proba = predict([content])[0]
                    prediction = "Fake ❌" if proba.argmax() == 0 else "Real ✅"

                    col1, col2 = st.columns(2)
                    col1.markdown(f"**prediction**: {prediction}")
                    col2.markdown(f"**Confidence**: {proba.max():.2%}")

                    # Explanation
                    st.markdown("### Explanation")
                    exp = explainer.explain_instance(
                        content,
                        predict,
                        num_features=10,
                    )

                    st.write(exp.as_list())

                    # Verified News Search
                    st.markdown("### Verified News")
                    Verified_news = newsapi.get_everything(
                        q = article['title'],
                        language = 'en',
                        sort_by='relevancy',
                        page_size=3
                    )

                    if Verified_news['articles']:
                        for v in Verified_news['articles']:
                            st.markdown(f"- [{v['title']}]({v['url']})")
                    else:
                        st.info("No verified news found for this topic.")
                    
                    st.markdown("---")
    except Exception as e:
        st.error(f"Error fetching news articles: {str(e)}")
else:
    st.info("Please enter a valid NewsAPI key to get started.")                
