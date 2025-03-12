import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime
import numpy as np

# Generate sample data if no file is uploaded
def generate_sample_data():
    """Generate sample training metrics data"""
    num_episodes = 1000
    data = {
        'episode': list(range(num_episodes)),
        'reward': [],
        'score': [],
        'epsilon': [],
        'avg_reward': [],
        'avg_score': [],
        'frames_alive': [],
        'best_score': []
    }
    
    # Generate synthetic data
    epsilon = 1.0
    for i in range(num_episodes):
        # Decay epsilon
        epsilon *= 0.997 if epsilon > 0.05 else 1.0
        
        # Generate synthetic metrics
        score = int(np.random.normal(i/100, 2))
        reward = score * 10 + np.random.normal(0, 10)
        frames = score * 100 + np.random.normal(0, 50)
        
        data['reward'].append(reward)
        data['score'].append(max(0, score))
        data['epsilon'].append(epsilon)
        data['frames_alive'].append(max(0, frames))
        data['best_score'].append(max(data['score']))
        
        # Calculate moving averages
        window = 10
        data['avg_reward'].append(np.mean(data['reward'][-window:]))
        data['avg_score'].append(np.mean(data['score'][-window:]))
    
    return pd.DataFrame(data)

def create_metric_chart(df, metric_name, avg_metric_name, color, title):
    """Create a plotly figure for a given metric"""
    fig = go.Figure()
    
    # Add raw data with low opacity
    fig.add_trace(
        go.Scatter(
            x=df['episode'],
            y=df[metric_name],
            name=f'Raw {metric_name.capitalize()}',
            line=dict(color=color, width=1),
            opacity=0.3
        )
    )
    
    # Add moving average
    fig.add_trace(
        go.Scatter(
            x=df['episode'],
            y=df[avg_metric_name],
            name=f'Average {metric_name.capitalize()}',
            line=dict(color=color, width=2)
        )
    )
    
    fig.update_layout(
        title=title,
        xaxis_title='Episode',
        yaxis_title=metric_name.capitalize(),
        hovermode='x unified',
        showlegend=True
    )
    
    return fig

def main():
    st.set_page_config(page_title="Flappy Bird AI Training Dashboard", layout="wide")
    
    st.title("Flappy Bird AI Training Metrics Dashboard")
    
    # Add radio button to choose between sample data and file upload
    data_source = st.radio(
        "Choose data source",
        ("Use sample data", "Upload training data")
    )
    
    if data_source == "Upload training data":
        uploaded_file = st.file_uploader("Upload training metrics JSON file", type=['json'])
        if uploaded_file is not None:
            df = pd.DataFrame(json.load(uploaded_file))
        else:
            st.info("Please upload a training metrics JSON file or switch to sample data")
            return
    else:
        df = generate_sample_data()
        st.info("Using sample data. To view your own training results, select 'Upload training data' and upload your JSON file.")
    
    # Summary metrics in the sidebar
    st.sidebar.header("Training Summary")
    st.sidebar.metric("Total Episodes", len(df))
    st.sidebar.metric("Best Score", df['best_score'].max())
    st.sidebar.metric("Final Epsilon", f"{df['epsilon'].iloc[-1]:.4f}")
    
    # Add date range selector
    st.sidebar.header("Episode Range")
    episode_range = st.sidebar.slider(
        "Select Episode Range",
        min_value=int(df['episode'].min()),
        max_value=int(df['episode'].max()),
        value=(int(df['episode'].min()), int(df['episode'].max()))
    )
    
    # Filter data based on selected range
    mask = (df['episode'] >= episode_range[0]) & (df['episode'] <= episode_range[1])
    filtered_df = df[mask]
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Performance Metrics", "Learning Progress", "Detailed Analysis"])
    
    with tab1:
        st.header("Game Performance")
        
        # Score metrics
        col1, col2 = st.columns(2)
        
        with col1:
            score_fig = create_metric_chart(
                filtered_df, 
                'score', 
                'avg_score',
                'rgb(239, 85, 59)',
                'Game Scores Over Time'
            )
            st.plotly_chart(score_fig, use_container_width=True)
            
        with col2:
            frames_fig = create_metric_chart(
                filtered_df,
                'frames_alive',
                'frames_alive',
                'rgb(99, 110, 250)',
                'Survival Time (Frames) Over Time'
            )
            st.plotly_chart(frames_fig, use_container_width=True)
    
    with tab2:
        st.header("Learning Progress")
        
        # Reward and epsilon metrics
        col1, col2 = st.columns(2)
        
        with col1:
            reward_fig = create_metric_chart(
                filtered_df,
                'reward',
                'avg_reward',
                'rgb(0, 204, 150)',
                'Rewards Over Time'
            )
            st.plotly_chart(reward_fig, use_container_width=True)
        
        with col2:
            # Epsilon decay
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=filtered_df['episode'],
                    y=filtered_df['epsilon'],
                    name='Epsilon',
                    line=dict(color='rgb(142, 68, 173)', width=2)
                )
            )
            fig.update_layout(
                title='Epsilon Decay Over Time',
                xaxis_title='Episode',
                yaxis_title='Epsilon',
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("Detailed Analysis")
        
        # Statistics for selected range
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Average Score",
                f"{filtered_df['score'].mean():.2f}",
                f"{filtered_df['score'].std():.2f} σ"
            )
        
        with col2:
            st.metric(
                "Average Reward",
                f"{filtered_df['reward'].mean():.2f}",
                f"{filtered_df['reward'].std():.2f} σ"
            )
        
        with col3:
            st.metric(
                "Average Survival Time",
                f"{filtered_df['frames_alive'].mean():.2f}",
                f"{filtered_df['frames_alive'].std():.2f} σ"
            )
        
        # Correlation heatmap
        st.subheader("Metric Correlations")
        numeric_cols = ['score', 'reward', 'frames_alive', 'epsilon']
        corr_matrix = filtered_df[numeric_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=numeric_cols,
            y=numeric_cols,
            colorscale='RdBu',
            zmin=-1,
            zmax=1,
            text=np.round(corr_matrix, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Correlation Matrix',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Raw data table
        st.subheader("Raw Data")
        st.dataframe(
            filtered_df.style.highlight_max(axis=0, subset=['score', 'reward', 'frames_alive']),
            height=300
        )

if __name__ == "__main__":
    main()