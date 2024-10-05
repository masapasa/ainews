# app.py
import streamlit as st
import pandas as pd
from processor import LanguageModelProcessor

def main():
    st.title("News Article Processor")
    
    # Initialize processor
    processor = LanguageModelProcessor()
    
    # Display data
    st.subheader("Articles Database")
    st.dataframe(processor.df)
    
    # Process articles button
    if st.button("Process Unprocessed Articles"):
        with st.spinner("Processing articles..."):
            processor.process_articles()
        st.success("Processing completed!")
        st.rerun()
    
    # Filter options
    st.subheader("Filter Options")
    filter_topic = st.multiselect("Filter by Topic", options=processor.df['topic'].unique())
    filter_processed = st.selectbox("Filter by Processing Status", 
                                  options=['All', 'Processed', 'Unprocessed'])
    
    # Apply filters
    filtered_df = processor.df.copy()
    if filter_topic:
        filtered_df = filtered_df[filtered_df['topic'].isin(filter_topic)]
    if filter_processed != 'All':
        if filter_processed == 'Processed':
            filtered_df = filtered_df[filtered_df['processed_language_model']]
        else:
            filtered_df = filtered_df[~filtered_df['processed_language_model']]
    
    # Display filtered results
    st.subheader("Filtered Results")
    st.dataframe(filtered_df)
    
    # Display article details
    st.subheader("Article Details")
    selected_id = st.selectbox("Select Article ID", options=filtered_df['id'].unique())
    if selected_id:
        article = filtered_df[filtered_df['id'] == selected_id].iloc[0]
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("Original Content")
            st.write("Title:", article['title'])
            st.write("Summary:", article['summary_detail'])
            st.write("Topic:", article['topic'])
            st.write("Labels:", article['labels'])
            
        with col2:
            st.write("AI Generated Content")
            if article['processed_language_model']:
                st.write("AI Summary:", article.get('ai_summary', 'Not available'))
                st.write("AI Keywords:", article.get('ai_keywords', 'Not available'))
            else:
                st.write("Article not yet processed")

if __name__ == "__main__":
    main()