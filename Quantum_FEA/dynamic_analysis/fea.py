""" 
Classical Preprocessing Stage:
This module loads FEA problem input, and returns Quantum Hamiltonian.

Equation of Motion:

M ddot X = - KX + F(t)

Here, M is the Mass Matrix of the structure, K is the stiffness matrix, and F(t)
is the external load input applied to nodes.

Case 1: Zero External Load

    M ddot X = - KX

    Reduction to  Quantum Evolution:

    
Case 2: With external load (static)
    M ddot X = -KX + F             eq(1)
    M ddot X = -K(X + K^{-1} F)
    Y = (X + K^{-1} F) 
    M ddot Y = -K Y

Case 3: With external load (dynamic)
    M ddot X = -KX + F(t)

    """



"""

alpha_i = Tr(P_iH)

H = 
[[0 , A]
A   0]]



{P_0 , alpha_0}, {P_1, alpha_1}

H = alph_0.P_0 + alpha_1.P_1 +..
"""



import scipy 
import pennylane as qml
import numpy as np 
np.random.seed(42)
from scipy.spatial import Delaunay
from scipy.linalg import sqrtm
from scipy.sparse import dok_matrix
import matplotlib.pyplot as plt 
from pennylane import ApproxTimeEvolution

dev = qml.device("default.qubit", wires=4)

def generate_symmetric_matrix(n):
    # Generate a random square matrix
    A = np.random.rand(n, n)
    # Make the matrix symmetric
    A_symmetric = (A + A.T) / 2
    return A_symmetric

def generate_convex_polygon(num_vertices):
    """
    Generates a set of points that form a convex polygon.

    The approach is to generate random points, calculate their centroid,
    sort them by angle from the centroid (which ensures convexity), and return
    the sorted points as the vertices of the convex polygon.
    """
    # Generate random points
    points = np.random.rand(num_vertices, 2) * 10

    # Calculate the centroid of the points
    centroid = np.mean(points, axis=0)

    # Calculate angles of points from the centroid
    angles = np.arctan2(points[:,1] - centroid[1], points[:,0] - centroid[0])

    # Sort points by angle
    sorted_points = points[np.argsort(angles)]

    return sorted_points


"""
Class Node:

Attributes: 

    index --------> node index
    x ------------> x coordinate of the node
    y ------------> y coordinate of the node
    F_x ----------> Force in x direction acting on the node
    F_y ----------> Force in y direction acting on the node
    b_x ----------> boundary condition in the x direction (1, if the x displacement of node is constrained, 0 otherwise )
    b_y ----------> boundary condition in the y direction (1, if the y displacement of node is constrained, 0 otherwise )
    adj_nodes ----> list containing indices of adjacent nodes

"""

class Node: 
    def __init__(self,index, x,y,F_x,F_y,b_x,b_y,adj_nodes):
        self.index = index 
        self.x = x
        self.y = y
        self.F_x = F_x
        self.F_y = F_y
        self.b_x = b_x
        self.b_y = b_y 
        self.adj_nodes = adj_nodes

    def assign_adjacent_nodes(self,adj_nodes):
        self.adj_nodes = adj_nodes 


class Structure:
    # definition: 
        # material properties
        # density, Youngs Modulus
    def __init__(self,n_nodes):

        # 1. Define Geometry

            # 1.1 define  points
        
        self.points = self._generate_geometry(n_nodes)
            # 1.2 Perform triangulation
    
        self.triangulation_vertices = Delaunay(self.points)

        # initialize list of nodes 

        self.nodes = [ Node(i,self.points[i][0], self.points[i][1],0,0,0,0,set([i])) for i in range(n_nodes)]

        # Define boundary Conditions:

        self._boundary_condition()

        # Assign adjacent nodes to each node
        for i in range(len(self.triangulation_vertices.simplices)):
            #print(self.triangulation_vertices.simplices[i])
            for j in range(3):
                adj = {x for x in self.triangulation_vertices.simplices[i] if x!= self.triangulation_vertices.simplices[i,j] }
                # print(adj)
                self.nodes[self.triangulation_vertices.simplices[i,j]].assign_adjacent_nodes(adj)
            
        # Compute Adjacency Matrix A
        self.A = self._compute_adjacency_matrix() # comment this out if N > 2**30 
        # Compute Stiffness Matrix K
        self.K = 10*self.A 
        # Define Mass Matrix M
        self.M = np.diag(np.full(len(self.nodes), 1.5))
        # Compute Hamiltonian
        H, H_block = self._compute_hamiltonian()
        self.H = H
        self.H_block = H_block
        # Compute F matrix. Note, M\ddot x = -K. \ddot x = -inv(M) K x => \ddot x = F x (F = -inv(M) K )
        self.F = self._compute_F()

    def _boundary_condition(self):
        # displacement boundary conditions:-
        # fix node 0 
        self.nodes[0].b_x = 1
        self.nodes[0].b_y = 1
        self.nodes[1].b_x = 1
        self.nodes[1].b_y = 1 
        # force boundary conditions:-

    
    def _generate_geometry(self, n_nodes):
        # generate random polygon 
        points = generate_convex_polygon(n_nodes)
        return points
    
    def _compute_adjacency_matrix(self):
        A = np.zeros((len(self.nodes), len(self.nodes)))
        for node in self.nodes:
            for j in node.adj_nodes:
                A[node.index,j] = 1
                # print("modidfied")
                A[j,node.index] = 1
        for j in range(len(self.nodes)):
            if self.nodes[j].b_x == 1:
                k_jj = 100
            else :
                k_jj = 0
            A[j,j] = -(np.sum(A[j]) + k_jj)
        return A 
    
        
    def visualize_geometry(self):
        
        """ 

        Method to visualize Results:
        
        """
        plt.figure(figsize=(6, 6))
        plt.triplot(self.points[:, 0], self.points[:, 1], self.triangulation_vertices.simplices.copy(), 'b-')
        plt.plot(self.points[:, 0], self.points[:, 1], 'o', markersize=5, color='red')
        plt.fill(self.points[:, 0], self.points[:, 1], alpha=0.2, color='grey')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title("Random Structure:")
        plt.show()

    def _compute_F(self):
        F = -np.linalg.inv(self.M) @ self.K
        return F 

    def _compute_hamiltonian(self):
        X = np.array([[0, 1], [1, 0]])
        H_squared =  - np.linalg.inv(np.sqrt(self.M)) @ self.K @ np.linalg.inv(np.sqrt(self.M))
        # returns quantum hamiltonian
        # print(H_squared)
        H = sqrtm(H_squared)
        H_block = np.kron(X, H)
        return H,H_block
    

if __name__ == '__main__':
    n = 2**6
    s_1 = Structure(n)
    print(s_1.points)
    print(s_1.points.shape)
    print("X coordinates: ",s_1.points[:,0])
    s_1.visualize_geometry()
    print("Adjoint Matrix:\n",s_1.A)
    print("Stiffness Matrix:\n",s_1.K)
    print("Mass Matrix:\n",s_1.M)
    print("Hamiltonian:\n", s_1.H)
    print(s_1.nodes[0].adj_nodes)
    H = s_1.H 
    H_block = s_1.H_block
    # save Hamiltonian Matrix on to file:
    np.save("H.npy", H)
    np.save("H_block.npy",H_block)
    is_symmetric = np.allclose(H, H.T)
    print(s_1.H_block)
    print(is_symmetric)
   

    




