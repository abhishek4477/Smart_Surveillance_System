import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

class CrowdAnalyzer:
    def __init__(self, csv_path: str):
        """Initialize crowd analyzer."""
        self.csv_path = csv_path
    
    def load_data(self, hours_back: int = None):
        """
        Load crowd data from CSV file.
        
        Args:
            hours_back: If specified, only load data from the last N hours
            
        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            df = pd.read_csv(self.csv_path)
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            
            if hours_back:
                cutoff_time = datetime.now() - timedelta(hours=hours_back)
                df = df[df['Timestamp'] >= cutoff_time]
            
            return df
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return pd.DataFrame()
    
    def create_time_series(self, hours_back: int = None):
        """Create time series plot of people count."""
        df = self.load_data(hours_back)
        if df.empty:
            return None
        
        fig = px.line(df, x='Timestamp', y='People_Count',
                     title='People Count Over Time',
                     labels={'People_Count': 'Number of People',
                            'Timestamp': 'Time'})
        
        fig.update_layout(
            showlegend=False,
            xaxis_title="Time",
            yaxis_title="Number of People",
            hovermode='x unified'
        )
        
        return fig
    
    def create_hourly_bar(self, hours_back: int = None):
        """Create bar chart of average people count by hour."""
        df = self.load_data(hours_back)
        if df.empty:
            return None
        
        df['Hour'] = df['Timestamp'].dt.hour
        hourly_avg = df.groupby('Hour')['People_Count'].mean().reset_index()
        
        fig = px.bar(hourly_avg, x='Hour', y='People_Count',
                    title='Average People Count by Hour',
                    labels={'People_Count': 'Average Number of People',
                           'Hour': 'Hour of Day'})
        
        fig.update_layout(
            showlegend=False,
            xaxis_title="Hour of Day",
            yaxis_title="Average Number of People",
            xaxis=dict(tickmode='linear', tick0=0, dtick=1)
        )
        
        return fig
    
    def create_heatmap(self, hours_back: int = None):
        """Create heatmap of people count by hour and day."""
        df = self.load_data(hours_back)
        if df.empty:
            return None
        
        df['Hour'] = df['Timestamp'].dt.hour
        df['Day'] = df['Timestamp'].dt.strftime('%Y-%m-%d')
        
        pivot_table = df.pivot_table(
            values='People_Count',
            index='Day',
            columns='Hour',
            aggfunc='mean'
        ).round(1)
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot_table.values,
            x=pivot_table.columns,
            y=pivot_table.index,
            colorscale='RdYlBu_r'
        ))
        
        fig.update_layout(
            title='People Count Heatmap',
            xaxis_title="Hour of Day",
            yaxis_title="Date",
            xaxis=dict(tickmode='linear', tick0=0, dtick=1)
        )
        
        return fig
    
    def get_current_stats(self):
        """Get current crowd statistics."""
        df = self.load_data(hours_back=24)  # Last 24 hours
        if df.empty:
            return {
                'current_count': 0,
                'daily_max': 0,
                'daily_avg': 0,
                'last_update': 'No data'
            }
        
        return {
            'current_count': df.iloc[-1]['People_Count'],
            'daily_max': df['People_Count'].max(),
            'daily_avg': round(df['People_Count'].mean(), 1),
            'last_update': df.iloc[-1]['Timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def check_overcrowding(self, threshold: int):
        """Check if current count exceeds threshold."""
        stats = self.get_current_stats()
        return stats['current_count'] > threshold 