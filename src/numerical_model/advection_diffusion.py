#!/usr/bin/env python3
"""
2D Advection-Diffusion Numerical Model for Contaminant Transport
Finite Differences Implementation

Description:
    This module implements a comprehensive 2D advection-diffusion model for contaminant transport
    in aquatic environments using finite differences. The model solves the governing equation:
    
    ∂C/∂t + u∂C/∂x + v∂C/∂y = D(∂²C/∂x² + ∂²C/∂y²) + S - kC
    
    Where:
    - C: contaminant concentration [mg/L]
    - u, v: advection velocities [m/s]
    - D: diffusion coefficient [m²/s]
    - S: source term [mg/(L·s)]
    - k: decay rate [1/s]
    
    Features:
    - Multiple boundary condition types (Dirichlet, Neumann, Mixed)
    - Numerical stability verification (CFL and diffusion conditions)
    - Flexible source term implementation with temporal control
    - Optimized finite difference schemes for accuracy and efficiency
    - Comprehensive result storage and analysis capabilities

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
from typing import Tuple, Dict, Any

# Third-party libraries
import numpy as np
import yaml

class AdvectionDiffusionModel:
    """
    2D advection-diffusion model using finite differences.
    """
    
    def __init__(self, config_path: str = None, config: dict = None):
        """
        Initialize the model with configuration parameters.
        
        Args:
            config_path: Path to YAML configuration file
            config: Configuration dictionary (alternative to config_path)
        """
        if config is not None:
            self.config = config
        elif config_path is not None:
            with open(config_path, 'r', encoding='utf-8') as file:
                self.config = yaml.safe_load(file)
        else:
            config_path = os.path.join(os.path.dirname(__file__), 
                                     '../../config/parameters.yaml')
            with open(config_path, 'r', encoding='utf-8') as file:
                self.config = yaml.safe_load(file)
        
        self._setup_domain()
        self._setup_physics()
        self._initialize_fields()
    
    def _setup_domain(self):
        """
        Configure spatial and temporal discretization parameters.
        
        This method initializes the computational domain by setting up the
        spatial grid (x, y coordinates) and temporal discretization based
        on the configuration parameters. It calculates grid dimensions and
        time step parameters for the finite difference scheme.
        
        Note:
            Grid points are calculated as (length / spacing) + 1 to include
            boundary points. The method also creates coordinate arrays for
            visualization and analysis purposes.
        """
        domain = self.config['domain']
        
        self.Lx = domain['length_x']
        self.Ly = domain['length_y']
        self.dx = domain['dx']
        self.dy = domain['dy']
        self.dt = domain['dt']
        self.total_time = domain['total_time']
        
        # Number of grid points
        self.nx = int(self.Lx / self.dx) + 1
        self.ny = int(self.Ly / self.dy) + 1
        self.nt = int(self.total_time / self.dt) + 1
        
        # Coordinates
        self.x = np.linspace(0, self.Lx, self.nx)
        self.y = np.linspace(0, self.Ly, self.ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        print(f"Domain: {self.Lx}m x {self.Ly}m")
        print(f"Grid: {self.nx} x {self.ny} points")
        print(f"Time steps: {self.nt}")
    
    def _setup_physics(self):
        """
        Configure physical parameters for the advection-diffusion equation.
        
        This method initializes the physical parameters that govern contaminant
        transport including diffusion coefficient, advection velocities, and
        decay rate. It also performs numerical stability checks to ensure
        the simulation will converge.
        
        Note:
            The method automatically validates CFL and diffusion stability
            conditions and raises warnings if the time step is too large
            for stable computation.
        """
        physics = self.config['physics']
        
        self.D = physics['diffusion_coefficient']
        self.u = physics['advection_velocity']['u']
        self.v = physics['advection_velocity']['v']
        self.k = physics['decay_rate']
        
        # Check numerical stability
        self._check_stability()
    
    def _check_stability(self):
        """
        Verify numerical stability conditions for the finite difference scheme.
        
        This method checks both CFL (Courant-Friedrichs-Lewy) condition for
        advection and diffusion stability condition to ensure the explicit
        finite difference scheme will produce stable and accurate results.
        
        Stability Conditions:
            - CFL condition: |u|*dt/dx ≤ 1 and |v|*dt/dy ≤ 1
            - Diffusion condition: D*dt*(1/dx² + 1/dy²) ≤ 0.5
            
        Raises:
            Warning: If stability conditions are violated, indicating potential
                   numerical instability or inaccurate results.
        """
        # CFL condition for advection
        cfl_x = abs(self.u) * self.dt / self.dx
        cfl_y = abs(self.v) * self.dt / self.dy
        
        # Stability condition for diffusion
        diff_x = self.D * self.dt / (self.dx**2)
        diff_y = self.D * self.dt / (self.dy**2)
        
        print(f"CFL x: {cfl_x:.3f}, CFL y: {cfl_y:.3f}")
        print(f"Diffusion x: {diff_x:.3f}, Diffusion y: {diff_y:.3f}")
        
        if cfl_x > 1.0 or cfl_y > 1.0:
            print("WARNING: CFL condition violated. Reduce dt.")
        
        if diff_x > 0.5 or diff_y > 0.5:
            print("WARNING: Stability condition for diffusion violated.")
    
    def _initialize_fields(self):
        """
        Initialize concentration fields and simulation history storage.
        
        This method creates the concentration arrays for current and next
        time steps, both initialized to zero (clean environment). It also
        sets up storage arrays for tracking the temporal evolution of the
        concentration field throughout the simulation.
        
        Note:
            Creates self.C (current concentration), self.C_new (next time step),
            self.concentration_history (snapshots), and self.time_points arrays.
        """
        self.C = np.zeros((self.ny, self.nx))
        self.C_new = np.zeros((self.ny, self.nx))
        
        # History for analysis
        self.concentration_history = []
        self.time_points = []
    
    def _apply_source(self, t: float) -> np.ndarray:
        """
        Apply contaminant source term to the computational domain.
        
        This method calculates the source term contribution at the current
        time step, applying contaminant injection at the specified location
        with the configured strength and duration. The source is active only
        during the specified time period.
        
        Args:
            t (float): Current simulation time in seconds
            
        Returns:
            np.ndarray: 2D array of source term values with shape (ny, nx),
                       where non-zero values indicate contaminant injection
                       
        Note:
            Source location is determined by the nearest grid point to the
            specified coordinates. Source strength is applied only if the
            current time is within the source duration period.
        """
        source_config = self.config['source']
        
        source_x = source_config['location']['x']
        source_y = source_config['location']['y']
        strength = source_config['strength']
        duration = source_config['duration']
        
        # Find indices closest to source
        i_source = int(source_x / self.dx)
        j_source = int(source_y / self.dy)
        
        source_term = np.zeros((self.ny, self.nx))
        
        # Apply source only during specified duration
        if t <= duration:
            # Distribute source over small area (3x3)
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    ii, jj = i_source + di, j_source + dj
                    if 0 <= ii < self.nx and 0 <= jj < self.ny:
                        source_term[jj, ii] = strength / 9.0  # Distribute uniformly
        
        return source_term
    
    def _apply_boundary_conditions(self):
        """
        Apply boundary conditions to the concentration field.
        
        This method enforces the specified boundary conditions on all four
        domain boundaries (left, right, bottom, top). It supports both
        Dirichlet (fixed concentration) and Neumann (fixed gradient)
        boundary conditions as configured in the parameters.
        
        Boundary Types:
            - Dirichlet: C = constant_value at boundary
            - Neumann: ∂C/∂n = constant_value at boundary
            
        Note:
            The method uses finite difference approximations for Neumann
            conditions and directly sets values for Dirichlet conditions.
            Mixed boundary conditions are supported.
        """
        bc_config = self.config['boundary_conditions']
        
        # Left boundary (x=0)
        if bc_config['left']['type'] == 'dirichlet':
            self.C_new[:, 0] = bc_config['left']['value']
        elif bc_config['left']['type'] == 'neumann':
            # Neumann condition: dC/dx = specified value
            # Using forward differences: (C[1] - C[0])/dx = value
            self.C_new[:, 0] = self.C_new[:, 1] - self.dx * bc_config['left']['value']
        
        # Right boundary (x=Lx)
        if bc_config['right']['type'] == 'dirichlet':
            self.C_new[:, -1] = bc_config['right']['value']
        elif bc_config['right']['type'] == 'neumann':
            # Using backward differences: (C[-1] - C[-2])/dx = value
            self.C_new[:, -1] = self.C_new[:, -2] + self.dx * bc_config['right']['value']
        
        # Bottom boundary (y=0)
        if bc_config['bottom']['type'] == 'dirichlet':
            self.C_new[0, :] = bc_config['bottom']['value']
        elif bc_config['bottom']['type'] == 'neumann':
            self.C_new[0, :] = self.C_new[1, :] - self.dy * bc_config['bottom']['value']
        
        # Top boundary (y=Ly)
        if bc_config['top']['type'] == 'dirichlet':
            self.C_new[-1, :] = bc_config['top']['value']
        elif bc_config['top']['type'] == 'neumann':
            self.C_new[-1, :] = self.C_new[-2, :] + self.dy * bc_config['top']['value']
    
    def _apply_neumann_boundaries(self):
        """
        Apply mixed boundary conditions (both Dirichlet and Neumann).
        
        This method handles mixed boundary condition configurations where
        different boundaries can have different types (Dirichlet or Neumann).
        It applies the appropriate boundary condition based on the configuration
        for each boundary.
        
        Boundary Types:
            - Dirichlet: C = constant_value at boundary
            - Neumann: ∂C/∂n = constant_value at boundary
            
        Note:
            This method is called for mixed boundary condition configurations
            where some boundaries use Neumann conditions while others may
            use Dirichlet conditions.
        """
        bc = self.config['boundary_conditions']
        
        if bc['type'] == 'mixed':
            # Left boundary
            if bc['types']['left'] == 'dirichlet':
                self.C_new[:, 0] = bc['values']['left']
            elif bc['types']['left'] == 'neumann':
                # ∂C/∂x = 0 at x=0 (free outlet)
                # Using finite differences: (C[:,1] - C[:,0]) / dx = 0
                # Therefore: C[:,0] = C[:,1]
                self.C_new[:, 0] = self.C_new[:, 1]
            
            # Right boundary
            if bc['types']['right'] == 'dirichlet':
                self.C_new[:, -1] = bc['values']['right']
            elif bc['types']['right'] == 'neumann':
                # ∂C/∂x = specified_value at x=Lx
                # Using finite differences: (C[:,-1] - C[:,-2]) / dx = value
                self.C_new[:, -1] = self.C_new[:, -2] + bc['values']['right'] * self.dx
            
            # Bottom boundary
            if bc['types']['bottom'] == 'dirichlet':
                self.C_new[0, :] = bc['values']['bottom']
            elif bc['types']['bottom'] == 'neumann':
                # ∂C/∂y = specified_value at y=0
                # Using finite differences: (C[1,:] - C[0,:]) / dy = value
                self.C_new[0, :] = self.C_new[1, :] - bc['values']['bottom'] * self.dy
            
            # Top boundary
            if bc['types']['top'] == 'dirichlet':
                self.C_new[-1, :] = bc['values']['top']
            elif bc['types']['top'] == 'neumann':
                # ∂C/∂y = specified_value at y=Ly
                # Using finite differences: (C[-1,:] - C[-2,:]) / dy = value
                self.C_new[-1, :] = self.C_new[-2, :] + bc['values']['top'] * self.dy
    
    def solve_timestep(self, t: float):
        """
        Solve one time step using explicit finite difference scheme.
        
        This method advances the concentration field by one time step using
        an explicit finite difference discretization of the 2D advection-diffusion
        equation. It applies upwind differencing for advection terms and central
        differencing for diffusion terms.
        
        Args:
            t (float): Current simulation time in seconds
            
        Note:
            The method updates the concentration field (self.C) and applies
            boundary conditions after the numerical update. Source terms are
            applied at the current time level.
        """
        # Apply source term
        source = self._apply_source(t)
        
        # Explicit finite difference scheme
        for j in range(1, self.ny - 1):
            for i in range(1, self.nx - 1):
                # Diffusion terms (second derivative)
                d2C_dx2 = (self.C[j, i+1] - 2*self.C[j, i] + self.C[j, i-1]) / (self.dx**2)
                d2C_dy2 = (self.C[j+1, i] - 2*self.C[j, i] + self.C[j-1, i]) / (self.dy**2)
                
                # Advection terms (first derivative, upwind)
                if self.u >= 0:
                    dC_dx = (self.C[j, i] - self.C[j, i-1]) / self.dx
                else:
                    dC_dx = (self.C[j, i+1] - self.C[j, i]) / self.dx
                
                if self.v >= 0:
                    dC_dy = (self.C[j, i] - self.C[j-1, i]) / self.dy
                else:
                    dC_dy = (self.C[j+1, i] - self.C[j, i]) / self.dy
                
                # Advection-diffusion equation
                self.C_new[j, i] = (self.C[j, i] + 
                                   self.dt * (self.D * (d2C_dx2 + d2C_dy2) - 
                                            self.u * dC_dx - self.v * dC_dy - 
                                            self.k * self.C[j, i] + source[j, i]))
        
        # Apply boundary conditions
        bc_config = self.config['boundary_conditions']
        if bc_config['type'] == 'mixed':
            self._apply_neumann_boundaries()
        else:
            self._apply_boundary_conditions()
        
        # Update field
        self.C = self.C_new.copy()
    
    def run_simulation(self, save_interval: int = 10) -> Dict[str, Any]:
        """
        Execute the complete 2D advection-diffusion simulation.
        
        This method runs the full temporal evolution of the contaminant transport
        simulation, solving the advection-diffusion equation at each time step
        and storing concentration snapshots for analysis and visualization.
        
        Args:
            save_interval (int, optional): Frequency of saving concentration
                snapshots. Defaults to 10 (save every 10 time steps).
                
        Returns:
            Dict[str, Any]: Simulation results containing:
                - 'concentration_history': List of 2D concentration arrays
                - 'time_points': Corresponding time values for each snapshot
                - 'x': Spatial x-coordinates array
                - 'y': Spatial y-coordinates array
                - 'final_concentration': Final concentration field
                - 'max_concentration': Maximum concentration reached
                - 'total_mass': Total contaminant mass in domain
                
        Note:
            Progress information is printed every 10% of simulation completion,
            showing current time, progress percentage, and maximum concentration.
        """
        print("Starting simulation...")
        
        # Time loop
        for n in range(self.nt):
            t = n * self.dt
            
            # Solve time step
            self.solve_timestep(t)
            
            # Save results every save_interval steps
            if n % save_interval == 0:
                self.concentration_history.append(self.C.copy())
                self.time_points.append(t)
                
                # Show progress
                if n % (self.nt // 10) == 0:
                    progress = (n / self.nt) * 100
                    max_conc = np.max(self.C)
                    print(f"Time: {t:.2f}s ({progress:.1f}%) - Maximum concentration: {max_conc:.2f}")
        
        # Save final state
        if (self.nt - 1) % save_interval != 0:
            self.concentration_history.append(self.C.copy())
            self.time_points.append(self.total_time)
        
        print("Simulation completed.")
        
        # Return results
        return {
            'concentration_history': self.concentration_history,
            'time_points': self.time_points,
            'x': self.x,
            'y': self.y,
            'X': self.X,
            'Y': self.Y,
            'final_concentration': self.C,
            'config': self.config
        }