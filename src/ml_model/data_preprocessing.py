#!/usr/bin/env python3
"""
Data Preprocessing Module for Machine Learning Applications
Feature Extraction and Dataset Preparation

Description:
    This module provides comprehensive data preprocessing capabilities for machine learning
    applications in contaminant transport analysis. It handles the transformation of
    numerical simulation results into structured datasets suitable for training
    classification and regression models.
    
    Key functionalities:
    - Feature extraction from spatio-temporal simulation data
    - Risk level classification based on concentration thresholds
    - Fundamental and derived feature set generation
    - Data normalization and standardization
    - Train-test dataset splitting with stratification
    - Hydrodynamic parameter calculation (Péclet numbers, travel times)
    - Spatial gradient computation for enhanced feature sets

All the codes presented below were developed by:
    Dr. Gerardo Tinoco Guerrero
    Universidad Michoacana de San Nicolás de Hidalgo
    gerardo.tinoco@umich.mx

With the funding of:
    Secretary of Science, Humanities, Technology and Innovation, SECIHTI (Secretaria de Ciencia, Humanidades, Tecnología e Innovación). México.
    Coordination of Scientific Research, CIC-UMSNH (Coordinación de la Investigación Científica de la Universidad Michoacana de San Nicolás de Hidalgo, CIC-UMSNH). México
    Aula CIMNE-Morelia. México
    SIIIA-MATH: Soluciones de Ingeniería. México

Date:
    February, 2025.

Last Modification:
    August, 2025.
"""

# Standard libraries
import os
import pickle
from typing import Tuple, List, Dict, Any

# Third-party libraries
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    """
    Class for preprocessing simulation data and creating ML datasets.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the preprocessor.
        
        Args:
            config_path: Path to configuration file
        """
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 
                                     '../../config/parameters.yaml')
        
        with open(config_path, 'r', encoding='utf-8') as file:
            self.config = yaml.safe_load(file)
        
        self.risk_thresholds = self.config['risk_thresholds']
        self.scaler = StandardScaler()
    
    def load_simulation_data(self, simulation_path: str) -> Dict[str, Any]:
        """
        Load simulation data from pickle file.
        
        Args:
            simulation_path: Path to simulation file
            
        Returns:
            Dictionary with simulation data
        """
        with open(simulation_path, 'rb') as f:
            data = pickle.load(f)
        return data
    
    def classify_risk_level(self, concentration: float) -> int:
        """
        Classify risk level based on concentration.
        
        Args:
            concentration: Contaminant concentration (mg/L)
            
        Returns:
            Risk level: 0 (low), 1 (medium), 2 (high)
        """
        if concentration < self.risk_thresholds['low']:
            return 0  # Low risk
        elif concentration < self.risk_thresholds['medium']:
            return 1  # Medium risk
        else:
            return 2  # High risk
    
    def extract_features(self, simulation_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Extract comprehensive features from numerical simulation data.
        
        This method processes simulation results to create a feature matrix
        suitable for machine learning. It extracts spatial, temporal, and
        hydrodynamic features including position coordinates, velocity fields,
        Péclet numbers, travel times, and concentration values.
        
        Args:
            simulation_data (Dict[str, Any]): Dictionary containing simulation results
                with keys 'coordinates', 'concentration', 'parameters', and 'times'
            
        Returns:
            pd.DataFrame: DataFrame with extracted features including:
                - Spatial features: source_x, source_y, x_position, y_position
                - Temporal features: time_normalized, travel_time_x, travel_time_y
                - Hydrodynamic features: velocity_u, velocity_v, peclet_x, peclet_y
                - Physical features: diffusion_coeff, source_strength
                - Target: concentration, risk_level
                
        Note:
            The method automatically classifies risk levels based on concentration
            thresholds defined in the configuration file.
        """
        coords = simulation_data['coordinates']
        X, Y = coords['X'], coords['Y']
        
        # Get simulation parameters
        params = simulation_data['parameters']
        source_x = params['source']['location']['x']
        source_y = params['source']['location']['y']
        u = params['physics']['advection_velocity']['u']
        v = params['physics']['advection_velocity']['v']
        D = params['physics']['diffusion_coefficient']
        
        features_list = []
        concentrations = []
        
        # Process each saved temporal point
        for t_idx, concentration_field in enumerate(simulation_data['concentration_history']):
            time = simulation_data['time_points'][t_idx]
            
            # Process each spatial point
            for j in range(concentration_field.shape[0]):
                for i in range(concentration_field.shape[1]):
                    x_pos = X[j, i]
                    y_pos = Y[j, i]
                    concentration = concentration_field[j, i]
                    
                    # Calculate geometric features
                    distance_to_source = np.sqrt((x_pos - source_x)**2 + (y_pos - source_y)**2)
                    
                    # Relative direction to source
                    dx = x_pos - source_x
                    dy = y_pos - source_y
                    
                    # Hydrodynamic features
                    # Theoretical arrival time (pure advection)
                    if u != 0:
                        travel_time_x = dx / u if dx > 0 else 0
                    else:
                        travel_time_x = 0
                    
                    if v != 0:
                        travel_time_y = dy / v if dy > 0 else 0
                    else:
                        travel_time_y = 0
                    
                    # Local Péclet number
                    if D > 0:
                        peclet_x = abs(u) * distance_to_source / D
                        peclet_y = abs(v) * distance_to_source / D
                    else:
                        peclet_x = peclet_y = 0
                    
                    # Temporal features
                    time_normalized = time / params['domain']['total_time']
                    
                    # Feature vector
                    features = {
                        'x_position': x_pos,
                        'y_position': y_pos,
                        'time': time,
                        'time_normalized': time_normalized,
                        'distance_to_source': distance_to_source,
                        'dx_from_source': dx,
                        'dy_from_source': dy,
                        'travel_time_x': travel_time_x,
                        'travel_time_y': travel_time_y,
                        'peclet_x': peclet_x,
                        'peclet_y': peclet_y,
                        'velocity_u': u,
                        'velocity_v': v,
                        'diffusion_coeff': D,
                        'source_strength': params['source']['strength'],
                        'concentration': concentration
                    }
                    
                    features_list.append(features)
                    concentrations.append(concentration)
        
        df = pd.DataFrame(features_list)
        
        # Add risk labels
        df['risk_level'] = df['concentration'].apply(self.classify_risk_level)
        
        return df
    
    def add_spatial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add additional spatial features.
        
        Args:
            df: DataFrame with basic features
            
        Returns:
            DataFrame with additional spatial features
        """
        df_enhanced = df.copy()
        
        # Neighborhood features (average in spatial window)
        # This requires grouping by time and calculating local statistics
        
        # Approximate spatial gradients
        for time_point in df['time'].unique():
            time_mask = df['time'] == time_point
            time_data = df[time_mask].copy()
            
            # Sort by position to calculate gradients
            time_data = time_data.sort_values(['y_position', 'x_position'])
            
            # Calculate gradients (simplified)
            time_data['conc_gradient_x'] = time_data['concentration'].diff()
            time_data['conc_gradient_y'] = time_data.groupby('x_position')['concentration'].diff()
            
            # Update main DataFrame
            df_enhanced.loc[time_mask, 'conc_gradient_x'] = time_data['conc_gradient_x'].fillna(0)
            df_enhanced.loc[time_mask, 'conc_gradient_y'] = time_data['conc_gradient_y'].fillna(0)
        
        # Fill NaN values
        df_enhanced = df_enhanced.fillna(0)
        
        return df_enhanced
    
    def process_simulation_data(self, simulation_data: Dict[str, Any], scenario_name: str, 
                               use_fundamental_features: bool = False) -> pd.DataFrame:
        """
        Process data from an individual simulation to create a DataFrame.
        
        Args:
            simulation_data: Simulation data
            scenario_name: Scenario name
            use_fundamental_features: Whether to use only fundamental features
            
        Returns:
            DataFrame with extracted features
        """
        if use_fundamental_features:
            # Use only fundamental features
            df = self.extract_fundamental_features(simulation_data)
        else:
            # Use complete method with derived features
            df = self.extract_features(simulation_data)
            df = self.add_spatial_features(df)
        
        # Add scenario identifier
        df['scenario'] = scenario_name
        
        return df
    
    def _extract_point_features(self, x_pos: float, y_pos: float, concentration: float,
                               time_normalized: float, params: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract fundamental features for a single spatial-temporal point.
        
        Args:
            x_pos: X coordinate
            y_pos: Y coordinate  
            concentration: Concentration value
            time_normalized: Normalized time
            params: Simulation parameters
            
        Returns:
            Dictionary with fundamental features
        """
        source_x = params['source']['location']['x']
        source_y = params['source']['location']['y']
        u = params['physics']['advection_velocity']['u']
        v = params['physics']['advection_velocity']['v']
        source_strength = params['source']['strength']
        
        return {
            'source_x': source_x,
            'source_y': source_y,
            'velocity_u': u,
            'velocity_v': v,
            'source_strength': source_strength,
            'x_position': x_pos,
            'y_position': y_pos,
            'time_normalized': time_normalized,
            'concentration': concentration
        }

    def extract_fundamental_features(self, simulation_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Extract only fundamental input parameters as features.
        
        This method creates a simplified feature set using only the basic
        input parameters without derived hydrodynamic calculations. This
        approach tests whether complex feature engineering improves model
        performance compared to using raw input parameters.
        
        Args:
            simulation_data (Dict[str, Any]): Dictionary containing simulation results
                with keys 'coordinates', 'concentration', 'parameters', and 'times'
            
        Returns:
            pd.DataFrame: DataFrame with fundamental features including:
                - Source position: source_x, source_y
                - Velocity field: velocity_u, velocity_v
                - Source characteristics: source_strength
                - Spatial coordinates: x_position, y_position
                - Temporal information: time_normalized
                - Target: concentration, risk_level
                
        Note:
            This feature set contains 8 fundamental parameters compared to
            16 features in the complete feature extraction method.
        """
        coords = simulation_data['coordinates']
        X, Y = coords['X'], coords['Y']
        params = simulation_data['parameters']
        
        features_list = []
        
        # Process each saved temporal point
        for t_idx, concentration_field in enumerate(simulation_data['concentration_history']):
            time = simulation_data['time_points'][t_idx]
            time_normalized = time / params['domain']['total_time']
            
            # Process each spatial point
            for j in range(concentration_field.shape[0]):
                for i in range(concentration_field.shape[1]):
                    x_pos = X[j, i]
                    y_pos = Y[j, i]
                    concentration = concentration_field[j, i]
                    
                    # Extract features using auxiliary function
                    features = self._extract_point_features(x_pos, y_pos, concentration, 
                                                           time_normalized, params)
                    features_list.append(features)
        
        df = pd.DataFrame(features_list)
        
        # Add risk labels
        df['risk_level'] = df['concentration'].apply(self.classify_risk_level)
        
        return df

    def prepare_ml_dataset(self, df: pd.DataFrame, 
                          target_column: str = 'risk_level',
                          use_fundamental_features: bool = False) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare and normalize dataset for machine learning training.
        
        This method selects appropriate features based on the specified approach,
        normalizes the feature matrix using StandardScaler, and prepares the
        target vector for classification tasks.
        
        Args:
            df (pd.DataFrame): DataFrame containing extracted features and target variable
            target_column (str, optional): Name of the target column. Defaults to 'risk_level'.
            use_fundamental_features (bool, optional): If True, uses only fundamental
                input parameters (8 features). If False, uses complete derived
                features (16 features). Defaults to False.
            
        Returns:
            Tuple[np.ndarray, np.ndarray, List[str]]: A tuple containing:
                - X: Normalized feature matrix of shape (n_samples, n_features)
                - y: Target vector of shape (n_samples,)
                - feature_names: List of selected feature names
                
        Note:
            The scaler is fitted on the provided data and stored in self.scaler
            for consistent normalization of future data.
        """
        if use_fundamental_features:
            # Use only fundamental features from parameters
            feature_columns = [
                'source_x', 'source_y',
                'velocity_u', 'velocity_v', 
                'source_strength',
                'x_position', 'y_position', 
                'time_normalized'
            ]
        else:
            # Use derived features (original method)
            feature_columns = [
                'x_position', 'y_position', 'time_normalized',
                'distance_to_source', 'dx_from_source', 'dy_from_source',
                'travel_time_x', 'travel_time_y', 'peclet_x', 'peclet_y',
                'velocity_u', 'velocity_v', 'diffusion_coeff', 'source_strength',
                'conc_gradient_x', 'conc_gradient_y'
            ]
        
        # Filter columns that exist in the DataFrame
        available_features = [col for col in feature_columns if col in df.columns]
        
        X = df[available_features].values
        y = df[target_column].values
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y, available_features
    
    def split_dataset(self, X: np.ndarray, y: np.ndarray, 
                     test_size: float = None, random_state: int = None) -> Tuple:
        """
        Split dataset into training and test sets with stratified sampling.
        
        This method performs stratified train-test split to ensure balanced
        representation of all risk classes in both training and test sets,
        which is crucial for classification tasks with potentially imbalanced data.
        
        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features)
            y (np.ndarray): Target vector of shape (n_samples,)
            test_size (float, optional): Proportion of dataset for test set.
                                       If None, uses value from configuration file.
            random_state (int, optional): Random seed for reproducible splits.
                                        If None, uses value from configuration file.
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
                - X_train: Training feature matrix
                - X_test: Test feature matrix
                - y_train: Training target vector
                - y_test: Test target vector
                
        Note:
            Stratification ensures that the proportion of samples for each target
            class is preserved in both training and test sets.
        """
        if test_size is None:
            test_size = self.config['ml_parameters']['test_size']
        
        if random_state is None:
            random_state = self.config['ml_parameters']['random_state']
        
        return train_test_split(X, y, test_size=test_size, 
                               random_state=random_state, stratify=y)