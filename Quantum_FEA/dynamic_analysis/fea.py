""" 
Classical Preprocessing Stage:
This module loads FEA problem input, and returns Quantum Hamiltonian.

Equation of Motion:

M ddot X = - KX + F(t)

Here, M is the Mass Matrix of the structure, K is the stiffness matrix, and F(t)
is the external load input applied to nodes.

Case 1: Zero External Load

    M ddot X = - KX
    
Case 2: With external load (static)
    M ddot X = -KX + F             eq(1)
    M ddot X = -K(X + K^{-1} F)
    Y = (X + K^{-1} F) 
    M ddot Y = -K Y

Case 3: With external load (dynamic)
    M ddot X = -KX + F(t)

    """
import scipy 
import numpy as np 
from scipy.spatial import Delaunay
from scipy.linalg import sqrtm
from scipy.sparse import dok_matrix

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

    def __init__(self,n_nodes):

        # 1. Define Geometry

            # 1.1 define  points
        
        self.points = self._generate_geometry(n_nodes)
            # 1.2 Perform triangulation
    
        self.triangulation_vertices = Delaunay(self.points)

        # initialize list of nodes 

        self.nodes = [ Node(i,self.points[i][0], self.points[i][1],0,0,0,0,set([i])) for i in range(n_nodes)]

        # Assign adjacent nodes to each node
        for i in range(len(self.triangulation_vertices.simplices)):
            #print(self.triangulation_vertices.simplices[i])
            for j in range(3):
                adj = {x for x in self.triangulation_vertices.simplices[i] if x!= self.triangulation_vertices.simplices[i,j] }
                # print(adj)
                self.nodes[self.triangulation_vertices.simplices[i,j]].assign_adjacent_nodes(adj)
            

        # Compute Stiffness Matrix K
        
        # Compute square root of A matrix. A = sqrt(inv(M))K sqrt(M)
    
    def _generate_geometry(self, n_nodes):
        # generate random polygon 
        points = generate_convex_polygon(n_nodes)
        return points
    
        
    #def visualize()

    def compute_hamiltonian(self):

        H_squared =  - np.linalg.inv(np.sqrt(self.M)) @ self.K @ np.linalg.inv(np.sqrt(self.M))
        # returns quantum hamiltonian
        # print(H_squared)
        H = sqrtm(H_squared)

        return H
    

if __name__ == '__main__':
    # n = 2**10
    # M = np.identity(n)
    # K = generate_symmetric_matrix(n)
    # F = np.array([0,1])             
    # S_1 = Structure(M,K,F)
    # H_2 = S_1.compute_hamiltonian()
    # print("done computation")
    # print(H_2)
    n = 2**30
    s_1 = Structure(n)
    print(s_1.nodes[100].adj_nodes)
    




