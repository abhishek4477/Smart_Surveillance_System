import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CrowdAnalyzer:
    def __init__(self, csv_path: str):
        """
        Initialize crowd analyzer.
        
        Args:
            csv_path: Path to the CSV file containing crowd data
        """
        self.csv_path = Path(csv_path)
    
    def load_data(self, hours_back: int = 24) -> pd.DataFrame:
        """
        Load crowd data from CSV file.
        
        Args:
            hours_back: Number of hours of data to load
            
        Returns:
            DataFrame containing crowd data
        """
        try:
            if not self.csv_path.exists():
                logger.warning(f"Data file not found: {self.csv_path}")
                return pd.DataFrame()
            
            # Read CSV file
            df = pd.read_csv(self.csv_path)
            
            # Convert timestamp to datetime
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            
            # Filter for recent data
            if hours_back > 0:
                cutoff_time = datetime.now() - timedelta(hours=hours_back)
                df = df[df['Timestamp'] >= cutoff_time]
            
            return df.sort_values('Timestamp')
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return pd.DataFrame()
    
    def create_time_series(self, df: pd.DataFrame) -> go.Figure:
        """Create time series plot of people count."""
        fig = px.line(df, 
                     x='Timestamp', 
                     y='People_Count',
                     title='People Count Over Time',
                     labels={'People_Count': 'Number of People',
                            'Timestamp': 'Time'})
        
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Number of People",
            hovermode='x unified'
        )
        return fig
    
    def create_hourly_bar(self, df: pd.DataFrame) -> go.Figure:
        """Create hourly average bar chart."""
        df['Hour'] = df['Timestamp'].dt.hour
        hourly_avg = df.groupby('Hour')['People_Count'].mean().reset_index()
        
        fig = px.bar(hourly_avg,
                    x='Hour',
                    y='People_Count',
                    title='Average People Count by Hour',
                    labels={'People_Count': 'Average Number of People',
                           'Hour': 'Hour of Day'})
        
        fig.update_layout(
            xaxis_title="Hour of Day",
            yaxis_title="Average Number of People",
            xaxis=dict(tickmode='linear', tick0=0, dtick=1)
        )
        return fig
    
    def create_heatmap(self, df: pd.DataFrame) -> go.Figure:
        """Create heatmap of people count by hour and day."""
        df['Hour'] = df['Timestamp'].dt.hour
        df['Day'] = df['Timestamp'].dt.strftime('%Y-%m-%d')
        
        # Create pivot table
        pivot_data = df.pivot_table(
            values='People_Count',
            index='Day',
            columns='Hour',
            aggfunc='mean'
        ).fillna(0)
        
        fig = px.imshow(pivot_data,
                       title='People Count Heatmap',
                       labels=dict(x="Hour of Day", 
                                 y="Date", 
                                 color="People Count"),
                       aspect='auto')
        
        fig.update_layout(
            xaxis_title="Hour of Day",
            yaxis_title="Date",
            xaxis=dict(tickmode='linear', tick0=0, dtick=1)
        )
        return fig
    
    def get_current_stats(self, df: pd.DataFrame) -> dict:
        """
        Get current crowd statistics.
        
        Args:
            df: DataFrame containing crowd data
            
        Returns:
            Dictionary containing current statistics
        """
        try:
            if df.empty:
                return {
                    'current_count': 0,
                    'daily_max': 0,
                    'daily_avg': 0,
                    'last_update': 'No data'
                }
            
            current_count = df['People_Count'].iloc[-1]
            daily_max = df['People_Count'].max()
            daily_avg = df['People_Count'].mean()
            last_update = df['Timestamp'].iloc[-1].strftime('%Y-%m-%d %H:%M:%S')
            
            return {
                'current_count': current_count,
                'daily_max': daily_max,
                'daily_avg': daily_avg,
                'last_update': last_update
            }
            
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {
                'current_count': 0,
                'daily_max': 0,
                'daily_avg': 0,
                'last_update': 'Error'
            }
    
    def check_overcrowding(self, df: pd.DataFrame, threshold: int) -> bool:
        """
        Check if current crowd level exceeds threshold.
        
        Args:
            df: DataFrame containing crowd data
            threshold: Overcrowding threshold
            
        Returns:
            True if overcrowded, False otherwise
        """
        try:
            if df.empty:
                return False
            
            current_count = df['People_Count'].iloc[-1]
            return current_count >= threshold
            
        except Exception as e:
            logger.error(f"Error checking overcrowding: {str(e)}")
            return False
    
    def get_hourly_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get hourly crowd statistics.
        
        Args:
            df: DataFrame containing crowd data
            
        Returns:
            DataFrame with hourly statistics
        """
        try:
            if df.empty:
                return pd.DataFrame()
            
            df['Hour'] = df['Timestamp'].dt.hour
            hourly_stats = df.groupby('Hour')['People_Count'].agg([
                'mean',
                'max',
                'min',
                'count'
            ]).round(2)
            
            return hourly_stats
            
        except Exception as e:
            logger.error(f"Error getting hourly stats: {str(e)}")
            return pd.DataFrame()
    
    def get_daily_pattern(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get daily crowd patterns.
        
        Args:
            df: DataFrame containing crowd data
            
        Returns:
            DataFrame with daily patterns
        """
        try:
            if df.empty:
                return pd.DataFrame()
            
            df['Date'] = df['Timestamp'].dt.date
            df['Hour'] = df['Timestamp'].dt.hour
            
            daily_pattern = df.pivot_table(
                values='People_Count',
                index='Date',
                columns='Hour',
                aggfunc='mean'
            ).round(2)
            
            return daily_pattern
            
        except Exception as e:
            logger.error(f"Error getting daily pattern: {str(e)}")
            return pd.DataFrame() 