# processor.py
import logging
import pandas as pd
from config import Config
from openai import OpenAI

client = OpenAI(api_key=Config.OPENAI_API_KEY)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LanguageModelProcessor:
    def __init__(self):
        self.df = pd.read_csv(Config.CSV_FILE_PATH)
        self.df['processed_language_model'] = self.df.get('processed_language_model', False)
        
    def fetch_unprocessed_articles(self):
        unprocessed = self.df[~self.df['processed_language_model']].to_dict('records')
        logger.info(f"Found {len(unprocessed)} unprocessed articles.")
        return unprocessed

    def generate_summary(self, text):
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that summarizes articles.",
                    },
                    {
                        "role": "user",
                        "content": f"Please summarize the following article:\n\n{text}",
                    },
                ],
                max_tokens=150,
                temperature=0.5,
            )
            summary = response.choices[0].message.content.strip()
            logger.info("Generated summary.")
            return summary
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return None

    def extract_metadata_and_keywords(self, text, title):
        try:
            prompt = (
                "Extract the following metadata from the article text and title:\n"
                "- Keywords (comma-separated)\n"
                f"Title: {title}\n"
                f"Article Text: {text}"
            )
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an assistant that extracts metadata and generates keywords for articles.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=150,
                temperature=0.5,
            )
            keywords = response.choices[0].message.content.strip()
            logger.info("Generated keywords.")
            return keywords
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return None

    def update_article(self, article_id, summary, keywords):
        try:
            idx = self.df[self.df['id'] == article_id].index[0]
            self.df.at[idx, 'ai_summary'] = summary
            self.df.at[idx, 'ai_keywords'] = keywords
            self.df.at[idx, 'processed_language_model'] = True
            self.df.to_csv(Config.CSV_FILE_PATH, index=False)
            logger.info(f"Updated article '{article_id}' with summary and keywords.")
        except Exception as e:
            logger.error(f"Error updating article '{article_id}': {e}")

    def process_articles(self):
        articles = self.fetch_unprocessed_articles()
        for article in articles:
            try:
                text = article['summary_detail']
                title = article['title']
                
                if not text:
                    logger.warning(f"Article '{article['id']}' has no content. Skipping.")
                    continue
                    
                summary = self.generate_summary(text)
                if not summary:
                    logger.warning(f"Failed to generate summary for article '{article['id']}'. Skipping.")
                    continue
                    
                keywords = self.extract_metadata_and_keywords(text, title)
                if not keywords:
                    logger.warning(f"Failed to extract keywords for article '{article['id']}'. Skipping.")
                    continue
                    
                self.update_article(article['id'], summary, keywords)
            except Exception as e:
                logger.error(f"Error processing article '{article['id']}': {e}")
                
        logger.info("Language Model Processing completed.")