# ML for Physicists: Physics-Informed Neural Networks for Differential Equations

**Course:** CP301 — Development Engineering Project  
**Project Title:** ML for Physicists  
**Supervisor:** Dr. Shankhadeep Chakrabortty  
**Institution:** Indian Institute of Technology Ropar (IIT Ropar)

## Overview

This project explores the application of machine learning to physics-governed differential equations, with a particular focus on Physics-Informed Neural Networks (PINNs). Rather than relying exclusively on labelled training data, PINNs incorporate governing differential equations, initial conditions, and boundary conditions into the learning process through residual-based loss minimization.

The study investigates how neural networks can approximate solutions to differential equations and how their predictions compare with analytical and conventional numerical methods. The framework is explored across classical mechanics, quantum mechanics, singular differential equations, and black hole physics.

The project progresses from standard second-order differential equations and harmonic oscillators to hypergeometric equations and the radial wave equation associated with a BTZ black hole. Particular attention is given to the behaviour of solutions near singular points and physical boundaries.

## Objectives

- Apply machine learning techniques to solve physics-based differential equations.
- Formulate Physics-Informed Neural Networks using differential-equation residuals.
- Compare neural-network predictions with analytical and numerical solutions.
- Investigate the role of training, residual minimization, and physical constraints in improving solution accuracy.
- Study the application of PINNs to classical and quantum harmonic oscillators.
- Explore the treatment of singular differential equations using neural networks.
- Apply the framework to the hypergeometric form of the BTZ black hole wave equation.
- Investigate solution behaviour near singular points and boundaries.

## Methodology

### 1. Neural Networks for Differential Equations

A neural network approximates an unknown function \(y(x)\) through a parameterized representation \(y_\theta(x)\), where the trainable parameters include weights and biases.

For a general second-order linear differential equation,

\[
y''+p(x)y'+q(x)y=0,
\]

the network predicts the solution and automatic differentiation is used to calculate its derivatives.

The differential-equation residual is defined as

\[
R(x)=y_\theta''(x)+p(x)y_\theta'(x)+q(x)y_\theta(x).
\]

A residual-based loss function penalizes violations of the governing equation:

\[
\mathcal{L}_{\mathrm{res}}
=\frac{1}{N}\sum_{i=1}^{N}|R(x_i)|^2.
\]

Initial and boundary conditions can be incorporated through additional loss terms. Training adjusts the network parameters to reduce the residual and satisfy the specified physical constraints.

### 2. Quantum Harmonic Oscillator

The time-independent Schrödinger equation for the quantum harmonic oscillator provides a benchmark for testing the PINN framework.

The project considers the analytical wavefunctions, numerical solutions obtained using a finite-difference discretization, and neural-network approximations.

The finite-difference method converts the differential equation into a matrix eigenvalue problem. The PINN approach instead learns an approximate wavefunction by minimizing the governing-equation residual and enforcing the required constraints.

The comparison illustrates the potential and limitations of neural-network methods for quantum-mechanical eigenvalue problems.

### 3. Damped Harmonic Oscillator

The damped harmonic oscillator is used to examine whether a neural network can reproduce the evolution of a classical dynamical system.

Its governing equation is

\[
m\ddot{x}+c\dot{x}+kx=0,
\]

where \(m\) is the mass, \(c\) is the damping coefficient,
