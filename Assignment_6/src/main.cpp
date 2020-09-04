// This example is heavily based on the tutorial at https://open.gl

////////////////////////////////////////////////////////////////////////////////
// OpenGL Helpers to reduce the clutter
#include "helpers.h"
// GLFW is necessary to handle the OpenGL context
#include <GLFW/glfw3.h>
// Linear Algebra Library
#include <Eigen/Dense>
#include <Eigen/Geometry>
// STL headers
#include <chrono>
#include <iostream>
#include <fstream>

using namespace std;
using namespace Eigen;

#define PI 3.14159265

//
void initMesh();
void rotateMesh(double degree, int option);
void scaleMesh(double scaleFactor);
void clickDrag();
void deleteMesh();
void drawWireframe(int i);


////////////////////////////////////////////////////////////////////////////////
Program programGlobal;
//Mode indicator
int mode = 0;

struct Ray {
	Vector3f origin;
	Vector3f direction;
	Ray() { }
	Ray(Vector3f o, Vector3f d) : origin(o), direction(d) { }
};

bool checkIntersectTri(Ray ray, Vector3f a, Vector3f b, Vector3f c)
{
	Vector3f A = a;
	Vector3f B = b;
	Vector3f C = c;

	Matrix3f matrixVar;

	// Vector3d ray_direction = ray.direction - ray.origin;
	Vector3f ray_direction = ray.direction;

	matrixVar << A(0)-C(0),A(0)-B(0),ray_direction(0),A(1)-C(1),A(1)-B(1),ray_direction(1),A(2)-C(2),A(2)-B(2),ray_direction(2);
	
	Vector3f solution;

	solution << A(0)-ray.origin(0), A(1)-ray.origin(1), A(2)-ray.origin(2);

	Vector3f result = matrixVar.colPivHouseholderQr().solve(solution);

	if(result(0) >= 0 && result(1) >= 0 && result(0)+result(1) <= 1 && result(2) >= 0)
	{
		return true;
	}
	return false;
}

// Mesh object, with both CPU data (Eigen::Matrix) and GPU data (the VBOs)
struct Mesh {
	Eigen::MatrixXf V; // mesh vertices [3 x n]
	Eigen::MatrixXi F; // mesh triangles [3 x m]

	Eigen::MatrixXf V_flat;
	Eigen::MatrixXi F_flat;

	Eigen::MatrixXf normal;
	Eigen::MatrixXf normal_flat;

	Eigen::MatrixXf color;

	// VBO storing vertex position attributes
	VertexBufferObject V_vbo;

	// VBO storing vertex indices (element buffer)
	VertexBufferObject F_vbo;

	// VBO storing normals
	VertexBufferObject Normal_vbo;

	// VBO storing colors;
	VertexBufferObject Color_vbo;

	// VAO storing the layout of the shader program for the object 'bunny'
	VertexArrayObject vao;

	Eigen::Matrix4f model;

	Vector3f centroid;

	// Color
	Vector3f currentColor;

	double minY;
	double maxY;
	double maxX;
	double minX;
	double minZ;
	double maxZ;

	bool checkIntersect(double xpos, double ypos);
	void calculateCentroid();	
	void calculateMinMax();
	void initialResize();
	void convertFlatVertex();
	void calculateNormalVertex();
	void calculateNormalFlatVertex();
	void calculateColor();
	Vector4f getNormal(Vector3f a, Vector3f b, Vector3f c);
	Vector3f getNormalFlat(Vector3f a, Vector3f b, Vector3f c);
};

void Mesh::calculateCentroid()
{
	double xComp = 0;
	double yComp = 0;
	double zComp = 0;

	if(mode == 0 || mode == 1 || mode == 2)
	{
		for(int i = 0; i < V.cols(); i++)
		{
			Vector4f vertex;
			vertex << V.col(i)(0),V.col(i)(1), V.col(i)(2), 1;
			Vector4f result = model * vertex;
			xComp += result(0);
			yComp += result(1);
			zComp += result(2);
		}

		centroid << xComp/V.cols(), yComp/V.cols(), zComp/V.cols();
	}
	else
	{
		cout << "Calculate Centroid Phong" << endl;
		for(int i = 0; i < V_flat.cols(); i++)
		{
			Vector4f vertex;
			vertex << V_flat.col(i)(0),V_flat.col(i)(1), V_flat.col(i)(2), 1;
			Vector4f result = model * vertex;
			xComp += result(0);
			yComp += result(1);
			zComp += result(2);
		}

		centroid << xComp/V_flat.cols(), yComp/V_flat.cols(), zComp/V_flat.cols();
	}
	

	
}

void Mesh::calculateMinMax()
{
	double minYL = INT_MAX;
	double maxYL = -INT_MAX;
	double maxXL = -INT_MAX;
	double minXL = INT_MAX;
	double maxZL = -INT_MAX;
	double minZL = INT_MAX;

	for(int i = 0; i < V.cols(); i++)
	{
		if(minYL > V.col(i)(1))
		{
			minYL = V.col(i)(1);
		}

		if(minXL > V.col(i)(0))
		{
			minXL = V.col(i)(0);
		}

		if(maxXL < V.col(i)(0))
		{
			maxXL = V.col(i)(0);
		}

		if(maxYL < V.col(i)(1))
		{
			maxYL = V.col(i)(1);
		}

		if(maxZL < V.col(i)(2))
		{
			maxZL = V.col(i)(2);
		}

		if(minZL > V.col(i)(2))
		{
			minZL = V.col(i)(2);
		}

		minX = minXL;
		minY = minYL;
		maxX = maxXL;
		maxY = maxYL;

		minZ = minZL;
		maxZ = maxZL;
	}
}

bool Mesh::checkIntersect(double xpos, double ypos)
{
	Vector3f origin(xpos,ypos,1);
	Vector3f direction(0,0,-0.3);
	Ray ray(origin,direction);

	for(int i = 0; i < F.cols(); i++)
	{
		Vector4f aV;
		aV << V.col(F.col(i)(0)), 1;
		Vector4f bV;
		bV << V.col(F.col(i)(1)), 1;
		Vector4f cV;
		cV << V.col(F.col(i)(2)), 1;

		Vector4f resultA = model * aV;
		Vector4f resultB = model * bV;
		Vector4f resultC = model * cV;

		Vector3f a(resultA(0),resultA(1),resultA(2));
		Vector3f b(resultB(0),resultB(1),resultB(2));
		Vector3f c(resultC(0),resultC(1),resultC(2));

		if(checkIntersectTri(ray,a,b,c))
		{
			return true;
		}
	}

	return false;
}

void Mesh::initialResize()
{
	double diffX = maxX - minX;
	double diffY = maxY - minY;
	double diffZ = maxZ - minZ;

	double xScale = 1/diffX;
	double yScale = 1/diffY;
	double zScale = 1/diffZ;

	Matrix4f scale(4,4);
	scale << xScale,0,0,0,  0,yScale,0,0,  0,0,zScale,0  ,0,0,0,1;

	// Render this on GPU 
	model = scale * model;

	// Update this on V but never update the program with V buffer
	// for(int i = 0; i < V.cols(); i++)
	// {
	// 	Vector4f vertex(V.col(i)(0), V.col(i)(1), V.col(i)(2), 1);
	// 	Vector4f result = scale * vertex;

	// 	V.col(i) << result(0), result(1), result(2);
	// }

	calculateCentroid();
	calculateMinMax();

	// Translate every shape
	double xTrans = 0 - centroid(0);
	double yTrans = 0 - centroid(1);
	double zTrans = 0 -	centroid(2);

	Matrix4f transMatrix(4,4);
	transMatrix << 1,0,0,xTrans, 0,1,0,yTrans, 0,0,1,zTrans, 0,0,0,1;

	// Render this on GPU
	model = transMatrix * model;

	// Update this on V but never update the program with V buffer
	// for(int i = 0; i < V.cols(); i++)
	// {
	// 	Vector4f vertex(V.col(i)(0), V.col(i)(1), V.col(i)(2), 1);
	// 	Vector4f result = transMatrix * vertex;

	// 	V.col(i) << result(0), result(1), result(2);
	// }

	calculateCentroid();
	// calculateMinMax();
	
}

void Mesh::convertFlatVertex()
{
	MatrixXf newV(3, F.cols() * 3);
	MatrixXi newF(3, F.cols());

	for(int i = 0; i < F.cols(); i++)
	{
		Vector3f a = V.col(F.col(i)(0));
		Vector3f b = V.col(F.col(i)(1));
		Vector3f c = V.col(F.col(i)(2));

		newV.col(3 * i) = a;
		newV.col(3 * i + 1) = b;
		newV.col(3 * i + 2) = c;

		newF.col(i)(0) = 3 * i;
		newF.col(i)(1) = 3 * i + 1;
		newF.col(i)(2) = 3 * i + 2;
	}

	V_flat = newV;
	F_flat = newF;

	//cout << F_flat.transpose() << endl;
}

Vector4f Mesh::getNormal(Vector3f a, Vector3f b, Vector3f c)
{
	// cout << "Points: \n" << a.transpose() << "\n" << b.transpose() << "\n" << c.transpose() << endl;
	Vector3f ba;
	Vector3f ca;

	ba = b - a;
	ca = c - a;

	Vector3f normalS = ba.cross(ca);

	// cout << "Normal: " << normal.transpose() << endl;

	Vector4f result;
	result << normalS,1;


	// cout << "Inside: " << result.transpose() << endl;

	return result;
}

Vector3f Mesh::getNormalFlat(Vector3f a, Vector3f b, Vector3f c)
{
	Vector3f ab = a - b;
	Vector3f cb = c - b;

	Vector3f normalS = ab.cross(cb);

	return normalS;

}

void Mesh::calculateNormalVertex()
{
	MatrixXf newNormal(4,V.cols());
	newNormal.setZero();

	normal(3,V.cols());
	normal.setZero();

	for(int i = 0; i < F.cols(); i++)
	{
		Vector3f a = V.col(F.col(i)(0));
		Vector3f b = V.col(F.col(i)(1));
		Vector3f c = V.col(F.col(i)(2));

		Vector4f normalA = getNormal(a,b,c);
		
		newNormal.col(F.col(i)(0)) += normalA;
		newNormal.col(F.col(i)(1)) += normalA;
		newNormal.col(F.col(i)(2)) += normalA;
	}

	for(int i = 0; i < normal.cols(); i++)
	{
		newNormal.col(i)(0) = newNormal.col(i)(0)/newNormal.col(i)(3);
		newNormal.col(i)(1) = newNormal.col(i)(1)/newNormal.col(i)(3);
		newNormal.col(i)(2) = newNormal.col(i)(2)/newNormal.col(i)(3);

		normal.col(i) << newNormal.col(i);
	}

	normal = newNormal;

	// cout << newNormal.col(2) << endl;
	// cout << normal.col(2)(0) << endl;

	// cout << normal << endl;
}

void Mesh::calculateNormalFlatVertex()
{
	// cout << V_flat.cols() << " " << F_flat.cols() << endl;
	MatrixXf newNormal(3,V_flat.cols());
	newNormal.setConstant(1);

	for(int i = 0; i < F_flat.cols(); i++)
	{
		// cout << F_flat.col(i)(0) << " " << F_flat.col(i)(1) << " " << F_flat.col(i)(2) << endl;

		Vector3f a = V_flat.col(F_flat.col(i)(0));
		Vector3f b = V_flat.col(F_flat.col(i)(1));
		Vector3f c = V_flat.col(F_flat.col(i)(2));

		Vector3f normalA = getNormalFlat(a,b,c);

		newNormal.col(F_flat.col(i)(0)) = normalA;
		newNormal.col(F_flat.col(i)(1)) = normalA;
		newNormal.col(F_flat.col(i)(2)) = normalA;
	}

	// cout << newNormal.col(F_flat.col(0)(0)) << "\n" << newNormal.col(F_flat.col(0)(1)) << endl;
	normal_flat = newNormal;

	for(int i = 0; i < normal_flat.cols(); i++)
	{
		normal_flat.col(i) = -normal_flat.col(i).normalized();
	}

	// cout << normal_flat.transpose() << endl;

	// cout << V_flat.cols() << " " << F_flat.cols() << " " << normal_flat.cols() << endl;
}

void Mesh::calculateColor()
{
	if(mode == 0 || mode == 1)
	{
		MatrixXf newMat(3,V.cols());
		

		for(int i = 0; i < newMat.cols(); i++)
		{
			newMat.col(i) << currentColor(0), currentColor(1), currentColor(2);
		}

		// cout << newMat << endl;

		color = newMat;
	}
	else
	{
		MatrixXf newMat(3,V_flat.cols());
		color(3,V_flat.cols());

		for(int i = 0; i < newMat.cols(); i++)
		{
			newMat.col(i) << currentColor(0), currentColor(1), currentColor(2);
		}

		color = newMat;
	}

	// cout << color << endl;
	
}

// Mesh bunny;
vector<Mesh> meshList;
int indexSelect = -1;

//Click Drag
MatrixXf selectedPos;
Matrix4f selectedModel;
bool drag = false;


double xMousePos;
double yMousePos;
double xMouseClick;
double yMouseClick;

Matrix4f trans(4,4);

////////////////////////////////////////////////////////////////////////////////

// Read a triangle mesh from an off file
void load_off(const std::string &filename, Eigen::MatrixXf &V, Eigen::MatrixXi &F) {
	std::ifstream in(filename);
	std::string token;
	in >> token;
	int nv, nf, ne;
	in >> nv >> nf >> ne;
	V.resize(3, nv);
	F.resize(3, nf);
	for (int i = 0; i < nv; ++i) {
		in >> V(0, i) >> V(1, i) >> V(2, i);
	}
	for (int i = 0; i < nf; ++i) {
		int s;
		in >> s >> F(0, i) >> F(1, i) >> F(2, i);
		assert(s == 3);
	}
}

////////////////////////////////////////////////////////////////////////////////

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
	// Get viewport size (canvas in number of pixels)
	int width, height;
	glfwGetFramebufferSize(window, &width, &height);

	// Get the size of the window (may be different than the canvas size on retina displays)
	int width_window, height_window;
	glfwGetWindowSize(window, &width_window, &height_window);

	// Get the position of the mouse in the window
	double xpos, ypos;
	glfwGetCursorPos(window, &xpos, &ypos);

	// Deduce position of the mouse in the viewport
	double highdpi = (double) width / (double) width_window;
	xpos *= highdpi;
	ypos *= highdpi;

	// Convert screen position to the canonical viewing volume
	double xcan = ((xpos/double(width))*2)-1;
	double ycan = (((height-1-ypos)/double(height))*2)-1; // NOTE: y axis is flipped in glfw

	// cout << xcan << ycan << endl;
	// TODO: Ray-casting for object selection (Ex.3)
	if(action == 0)
	{
		bool found = true;
		// cout << xcan << " " << ycan << endl;
		for(int i = meshList.size() - 1; i >= 0; i--)
		{
			if(meshList.at(i).checkIntersect(xcan,ycan))
			{
				found = false;
				indexSelect = i;
				cout << "Select Mesh " << i << endl; 
				
			}
		}

		if(found)
		{
			indexSelect = -1;
		}
		else
		{ 
			if(mode == 0)
			{
				selectedPos = meshList.at(indexSelect).V;
				selectedModel = meshList.at(indexSelect).model;
			}
			else
			{
				selectedPos = meshList.at(indexSelect).V_flat;
				selectedModel = meshList.at(indexSelect).model;
			}
			
		}
	}

	if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if(GLFW_PRESS == action)
		{
			xMouseClick = xcan;
			yMouseClick = ycan;
			// std::cout << xMouseClick << " " << yMouseClick << std::endl;
            drag = true;
		}
        else if(GLFW_RELEASE == action)
		{
			// cout << "Release button" << endl;
			if(indexSelect != -1)
			{
				// cout << "Set selected" << endl;
				selectedModel = meshList.at(indexSelect).model;
				selectedPos = meshList.at(indexSelect).V;
			}
            drag = false;
		}
    }

	if(action == 2)
	{
		cout << "Release button" << endl;
	}
	
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	// Update the position of the first vertex if the keys 1,2, or 3 are pressed
	switch (key) {
		case GLFW_KEY_1:
			if(action == 0)
			{
				// cout << "Mesh cube" << endl;
				Mesh d;
				load_off(DATA_DIR "cube.off", d.V, d.F);
				meshList.push_back(d);
				initMesh();
			}
		break;
		case GLFW_KEY_2:
			if(action == 0)
			{
				Mesh d;
				load_off(DATA_DIR "bumpy_cube.off", d.V, d.F);
				meshList.push_back(d);
				initMesh();
			}
			break;
		case GLFW_KEY_3:
			if(action == 0)
			{
				Mesh d;
				load_off(DATA_DIR "bunny.off", d.V, d.F);
				meshList.push_back(d);
				initMesh();
			}
			break;
		case GLFW_KEY_4:
			if(action == 0)
			{
				Mesh d;
				load_off(DATA_DIR "dragon.off", d.V, d.F);
				meshList.push_back(d);
				initMesh();
			}
			break;
		case GLFW_KEY_Q:
			if(action == 0 && indexSelect != -1)
			{
				rotateMesh(10,1);
			}
			break;
		case GLFW_KEY_W:
			if(action == 0 && indexSelect != -1)
			{
				rotateMesh(-10,1);
			}
			break;
		case GLFW_KEY_R:
			if(action == 0 && indexSelect != -1)
			{
				rotateMesh(10,0);
			}
			break;
		case GLFW_KEY_E:
			if(action == 0 && indexSelect != -1)
			{
				rotateMesh(-10,0);
			}
			break;
		case GLFW_KEY_S:
			if(action == 0 && indexSelect != -1)
			{
				scaleMesh(1.1);
			}
			break;
		case GLFW_KEY_A:
			if(action == 0 && indexSelect != -1)
			{
				scaleMesh(0.9);
			}
			break;
		case GLFW_KEY_D:
			if(action == 0 && indexSelect != -1)
			{
				deleteMesh();
			}
			break;
		case GLFW_KEY_M:
			if(action == 0)
			{
				if(mode == 0)
				{
					cout << "Switch to mode 1: flat shading" << endl;
					
					mode = 1;
					// for(int i = 0; i < meshList.size(); i++)
					// {
					// 	meshList.at(i).V_vbo.update(meshList.at(i).V);
					// 	meshList.at(i).F_vbo.update(meshList.at(i).F);
					// }

					indexSelect = -1;
				}
				else if(mode == 1)
				{
					cout << "Switch to mode 2: phong shading" << endl;
					mode = 2;
					// for(int i = 0; i < meshList.size(); i++)
					// {
					// 	meshList.at(i).V_vbo.update(meshList.at(i).V_phong);
					// 	meshList.at(i).F_vbo.update(meshList.at(i).F_phong);
					// }

					indexSelect = -1;
				}
				else if(mode == 2)
				{
					cout << "Switch to mode 0: wireframe" << endl;
					mode = 0;
					// for(int i = 0; i < meshList.size(); i++)
					// {
					// 	meshList.at(i).V_vbo.update(meshList.at(i).V);
					// 	meshList.at(i).F_vbo.update(meshList.at(i).F);
					// }

					indexSelect = -1;
				}
			}
			break;
		case GLFW_KEY_7:
			if(action == 0 && indexSelect != -1)
			{
				meshList.at(indexSelect).currentColor << 0.4,0.4,0.4;
			}
			else
			{
				cout << "Please select/click an object first" << endl;
			}
			break;
		case GLFW_KEY_8:
			if(action == 0 && indexSelect != -1)
			{
				meshList.at(indexSelect).currentColor << 0.4,0.5,0.4;
			}
			else
			{
				cout << "Please select/click an object first" << endl;
			}
			break;
		case GLFW_KEY_9: 
			if(action == 0 && indexSelect != -1)
			{
				meshList.at(indexSelect).currentColor << 0.2,0.1,0.2;
			}
			else
			{
				cout << "Please select/click an object first" << endl;
			}
			break;
		case GLFW_KEY_0:
			if(action == 0 && indexSelect != -1)
			{
				meshList.at(indexSelect).currentColor << 0.2,0.7,0.2;
			}
			else
			{
				cout << "Please select/click an object first" << endl;
			}
			break;
		break;
		default:
			break;
	}
}

static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos)
{
	int width, height;
	glfwGetFramebufferSize(window, &width, &height);

	// Get the size of the window (may be different than the canvas size on retina displays)
	int width_window, height_window;
	glfwGetWindowSize(window, &width_window, &height_window);

	// Deduce position of the mouse in the viewport
	double highdpi = (double) width / (double) width_window;
	xpos *= highdpi;
	ypos *= highdpi;

	// Convert screen position to world coordinates
	double xworld = ((xpos/double(width))*2)-1;
	double yworld = (((height-1-ypos)/double(height))*2)-1; // NOTE: y axis is flipped in glfw
	
	xMousePos = xworld;
	yMousePos = yworld;

	// std::cout << xMousePos << " " << yMousePos << std::endl;
}

////////////////////////////////////////////////////////////////////////////////
//User programs

void initMesh()
{
	// Initialize the VBOs and others in new mesh
	int i = meshList.size() - 1;

	// Mesh newMesh = meshList.at(i);
	meshList.at(i).V_vbo.init(GL_FLOAT, GL_ARRAY_BUFFER);
	meshList.at(i).F_vbo.init(GL_UNSIGNED_INT, GL_ELEMENT_ARRAY_BUFFER);
	meshList.at(i).Normal_vbo.init(GL_FLOAT, GL_ARRAY_BUFFER);
	meshList.at(i).Color_vbo.init(GL_FLOAT, GL_ARRAY_BUFFER);

	meshList.at(i).convertFlatVertex();

	meshList.at(i).model(4,4);
	meshList.at(i).model << 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1;

	meshList.at(i).calculateCentroid();

	meshList.at(i).calculateNormalVertex();
	meshList.at(i).calculateNormalFlatVertex();

	meshList.at(i).currentColor << 0.4,0.4,0.4;
	meshList.at(i).calculateColor();

	// Vertex positions
	meshList.at(i).V_vbo.update(meshList.at(i).V);

	// Triangle indices
	meshList.at(i).F_vbo.update(meshList.at(i).F);

	// Update Normal indices
	meshList.at(i).Normal_vbo.update(meshList.at(i).normal);

	// Update color 
	meshList.at(i).Color_vbo.update(meshList.at(i).color);

	meshList.at(i).calculateMinMax();
	meshList.at(i).initialResize();

	// Create a new VAO for the bunny. and bind it
	meshList.at(i).vao.init();
	meshList.at(i).vao.bind();


	// Bind the element buffer, this information will be stored in the current VAO
	meshList.at(i).F_vbo.bind();

	// The vertex shader wants the position of the vertices as an input.
	// The following line connects the VBO we defined above with the position "slot"
	// in the vertex shader
	programGlobal.bindVertexAttribArray("position", meshList.at(i).V_vbo);
	programGlobal.bindVertexAttribArray("normal", meshList.at(i).Normal_vbo);
	programGlobal.bindVertexAttribArray("color", meshList.at(i).Color_vbo);

	// Unbind the VAO when I am done
	meshList.at(i).vao.unbind();

}

void rotateMesh(double degree, int option)
{
	double rotDeg = (degree*PI/180);

	double xtrans = 0 - meshList.at(indexSelect).centroid(0);
	double ytrans = 0 - meshList.at(indexSelect).centroid(1);
	double ztrans = 0 - meshList.at(indexSelect).centroid(2);
	
	Matrix4Xf rotateMat(4,4);
	Matrix4Xf transMat(4,4);
	Matrix4Xf transBackMat(4,4);
	// Matrix4Xf rotateZ(4,4);

	//0 Rotate y
	//1 Rotate X
	//2 Rotate Z
	if(option == 0)
		rotateMat << cos(rotDeg),0,sin(rotDeg),0, 0,1,0,0, -sin(rotDeg),0,cos(rotDeg),0, 0,0,0,1;
	else if(option == 1)
		rotateMat << 1,0,0,0, 0,cos(rotDeg),-sin(rotDeg),0, 0,sin(rotDeg),cos(rotDeg),0, 0,0,0,1;
	else if(option == 2)
		rotateMat << cos(rotDeg),-sin(rotDeg),0,0, sin(rotDeg),cos(rotDeg),0,0, 0,0,1,0, 0,0,0,1;
	else
		cout << "An error has occurred, please check code" << endl;
	
	// rotateZ << 1,0,0,0, 0,cos(rotDeg),-sin(rotDeg),0, 0,sin(rotDeg),cos(rotDeg),0, 0,0,0,1;
	// rotateMat << cos(rotDeg),0,sin(rotDeg),0, 0,1,0,0, -sin(rotDeg),0,cos(rotDeg),0, 0,0,0,1;
	transMat << 1,0,0,xtrans, 0,1,0,ytrans, 0,0,1,ztrans, 0,0,0,1;
	transBackMat << 1,0,0,-xtrans, 0,1,0,-ytrans, 0,0,1,-ztrans, 0,0,0,1;

	// Render on GPU
	meshList.at(indexSelect).model = transBackMat * rotateMat * transMat * meshList.at(indexSelect).model;

	meshList.at(indexSelect).calculateCentroid();
	meshList.at(indexSelect).calculateMinMax();

	selectedModel = meshList.at(indexSelect).model;
}

void scaleMesh(double scaleFactor)
{
	double xtrans = 0 - meshList.at(indexSelect).centroid(0);
	double ytrans = 0 - meshList.at(indexSelect).centroid(1);
	double ztrans = 0 - meshList.at(indexSelect).centroid(2);

	double xScale = scaleFactor;
	double yScale = scaleFactor;
	double zScale = scaleFactor;
	
	Matrix4Xf scale(4,4);
	Matrix4Xf transMat(4,4);
	Matrix4Xf transBackMat(4,4);
	// rotateMat << cos(rotDeg),0,sin(rotDeg),0, 0,1,0,0, -sin(rotDeg),0,cos(rotDeg),0, 0,0,0,1;
	scale << xScale,0,0,0,  0,yScale,0,0,  0,0,zScale,0  ,0,0,0,1;
	transMat << 1,0,0,xtrans, 0,1,0,ytrans, 0,0,1,ztrans, 0,0,0,1;
	transBackMat << 1,0,0,-xtrans, 0,1,0,-ytrans, 0,0,1,-ztrans, 0,0,0,1;

	// Render on GPU
	meshList.at(indexSelect).model = transBackMat * scale * transMat * meshList.at(indexSelect).model;

	meshList.at(indexSelect).calculateCentroid();
	meshList.at(indexSelect).calculateMinMax();

	for(int i = 0; i < selectedPos.cols(); i++)
	{
		Vector4f vertex(meshList.at(indexSelect).V.col(i)(0),meshList.at(indexSelect).V.col(i)(1), meshList.at(indexSelect).V.col(i)(2), 1);
		Vector4f result = meshList.at(indexSelect).model * vertex;
		selectedPos.col(i) << result(0),result(1),result(2); 
	}

	selectedModel = meshList.at(indexSelect).model;
}

void clickDrag()
{
	if(drag && indexSelect != -1)
	{
		Mesh curr = meshList.at(indexSelect);
		double diffX = (xMousePos - xMouseClick);
		double diffY = (yMousePos - yMouseClick);

		Matrix4f transMat(4,4);
		transMat << 1,0,0,diffX, 0,1,0,diffY, 0,0,1,0, 0,0,0,1;

		// Send to GPU
		meshList.at(indexSelect).model = transMat * selectedModel;

		meshList.at(indexSelect).calculateCentroid();
		meshList.at(indexSelect).calculateMinMax();

		
	}
}

void deleteMesh()
{
	if(indexSelect != -1)
	{
		meshList.at(indexSelect).vao.free();
		meshList.at(indexSelect).V_vbo.free();
		meshList.at(indexSelect).F_vbo.free();

		// Remove from meshList
		int endIndex = meshList.size() - 1;
		meshList.erase(meshList.begin() + indexSelect);

		indexSelect = -1;
	}



}

void drawWireframe(int i)
{
	Eigen::MatrixXi temp(3,1);
	if(mode == 0)
	{
		for(int j = 0; j < meshList.at(i).F.cols(); j++)
		{
			temp << meshList.at(i).F.col(j)(0), meshList.at(i).F.col(j)(1), meshList.at(i).F.col(j)(2);
			meshList.at(i).F_vbo.update(temp);

			glDrawElements(GL_LINE_LOOP, 3, meshList.at(i).F_vbo.scalar_type, 0);
		}
	}
	else
	{
		for(int z = 0; z < meshList.at(i).F_flat.cols(); z++)
		{
			temp << meshList.at(i).F_flat.col(z)(0), meshList.at(i).F_flat.col(z)(1), meshList.at(i).F_flat.col(z)(2);
			meshList.at(i).F_vbo.update(temp);

			glDrawElements(GL_LINE_LOOP, 3, meshList.at(i).F_vbo.scalar_type, 0);
		}
	}
}

////////////////////////////////////////////////////////////////////////////////

int main(void) {

	// Initialize the GLFW library
	if (!glfwInit()) {
		return -1;
	}

	// Activate supersampling
	glfwWindowHint(GLFW_SAMPLES, 8);

	// Ensure that we get at least a 3.2 context
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);

	// On apple we have to load a core profile with forward compatibility
#ifdef __APPLE__
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

	// Create a windowed mode window and its OpenGL context
	GLFWwindow * window = glfwCreateWindow(640, 640, "[Float] Hello World", NULL, NULL);
	if (!window) {
		glfwTerminate();
		return -1;
	}

	// Make the window's context current
	glfwMakeContextCurrent(window);

	// Load OpenGL and its extensions
	if (!gladLoadGL()) {
		printf("Failed to load OpenGL and its extensions");
		return(-1);
	}
	printf("OpenGL Version %d.%d loaded", GLVersion.major, GLVersion.minor);

	int major, minor, rev;
	major = glfwGetWindowAttrib(window, GLFW_CONTEXT_VERSION_MAJOR);
	minor = glfwGetWindowAttrib(window, GLFW_CONTEXT_VERSION_MINOR);
	rev = glfwGetWindowAttrib(window, GLFW_CONTEXT_REVISION);
	printf("OpenGL version recieved: %d.%d.%d\n", major, minor, rev);
	printf("Supported OpenGL is %s\n", (const char*)glGetString(GL_VERSION));
	printf("Supported GLSL is %s\n", (const char*)glGetString(GL_SHADING_LANGUAGE_VERSION));

	// Initialize the OpenGL Program
	// A program controls the OpenGL pipeline and it must contains
	// at least a vertex shader and a fragment shader to be valid
	// Program program;
	const GLchar* vertex_shader = R"(
		#version 150 core

		uniform mat4 model;
		uniform mat4 view;
		uniform mat4 proj;

		in vec3 normal;
		in vec3 position;
		in vec3 color;

		out vec3 Normal;
		out vec3 Position;
		out vec3 Color;

		void main() {
			mat4 temp = transpose(inverse(model));
			Normal = vec3(temp * vec4(normal, 1.0));
			Position = vec3(model * vec4(position,1.0));
			Color = color;
			gl_Position = proj * view * model * vec4(position, 1.0);
		}
	)";

	const GLchar* fragment_shader = R"(
		#version 150 core

		uniform vec3 light;
		out vec4 outColor;

		in vec3 Normal;
		in vec3 Position;
		in vec3 Color;

		void main() {
			vec3 ray_to_light = normalize(light - Position);
			vec3 norm = -normalize(Normal);
			float angleValue = max(dot(norm, ray_to_light), 0.0);

			vec3 diffuse = angleValue * Color;

			outColor = vec4(0.8 * diffuse, 1.0);
		}
	)";

	// Compile the two shaders and upload the binary to the GPU
	// Note that we have to explicitly specify that the output "slot" called outColor
	// is the one that we want in the fragment buffer (and thus on screen)
	programGlobal.init(vertex_shader, fragment_shader, "outColor");

	trans << 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1;

	//Load initial bunny
	Mesh bunnyInitial;
	load_off(DATA_DIR "bunny.off", bunnyInitial.V, bunnyInitial.F);
	meshList.push_back(bunnyInitial);

	
	// cout << meshList.at(0).V.cols() << " " << meshList.at(0).F.cols() << endl;

	// Prepare a dummy bunny object
	// We need to initialize and fill the two VBO (vertex positions + indices),
	// and use a VAO to store their layout when we use our shader program later.
	initMesh();

	// For the first exercises, 'view' and 'proj' will be the identity matrices
	// However, the 'model' matrix must change for each model in the scene
	Eigen::Matrix4f I = Eigen::Matrix4f::Identity();
	programGlobal.bind();
	glUniformMatrix4fv(programGlobal.uniform("view"), 1, GL_FALSE, I.data());
	glUniformMatrix4fv(programGlobal.uniform("proj"), 1, GL_FALSE, I.data());

	glUniform3f(programGlobal.uniform("light"), 0,0,1);

	// Save the current time --- it will be used to dynamically change the triangle color
	auto t_start = std::chrono::high_resolution_clock::now();

	// Register the keyboard callback
	glfwSetKeyCallback(window, key_callback);

	// Register the mouse callback
	glfwSetMouseButtonCallback(window, mouse_button_callback);

	// Set continous position callback
	glfwSetCursorPosCallback(window, cursor_position_callback);

	// Loop until the user closes the window
	while (!glfwWindowShouldClose(window)) {
		// Set the size of the viewport (canvas) to the size of the application window (framebuffer)
		int width, height;
		glfwGetFramebufferSize(window, &width, &height);
		glViewport(0, 0, width, height);

		// Clear the framebuffer
		glClearColor(0.5f, 0.5f, 0.5f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT  | GL_DEPTH_BUFFER_BIT);

		glEnable(GL_DEPTH_TEST);  

		// Bind your program
		programGlobal.bind();

		for(int i = 0; i < meshList.size(); i++)
		{
			// cout << meshList.size() << endl;
			// Mesh stuff = meshList.at(i);
			// Bind the VAO for the bunny
			meshList.at(i).vao.bind();

			// Model matrix for the bunny
			// glUniformMatrix4fv(programGlobal.uniform("model"), 1, GL_FALSE, I.data());

			// Set the uniform value depending on the time difference
			auto t_now = std::chrono::high_resolution_clock::now();
			float time = std::chrono::duration_cast<std::chrono::duration<float>>(t_now - t_start).count();
			// (float)(sin(time * 4.0f) + 1.0f) / 2.0f, 0.0f, 0.0f
			// glUniform3f(programGlobal.uniform("triangleColor"), 0,0,0);

			// Matrix4f trans(4,4);
			// trans << 1,0,0,0.1, 0,1,0,0.1, 0,0,1,0, 0,0,0,1;
			Matrix4f current = meshList.at(i).model;
			glUniformMatrix4fv(programGlobal.uniform("model"), 1, GL_FALSE, current.data());

			// Draw the triangles
			// GL_LINE_LOOP
			// glDrawElements(GL_TRIANGLES, 3 * meshList.at(i).F.cols(), meshList.at(i).F_vbo.scalar_type, 0);
			// glDrawElements(GL_LINE_LOOP, 3 * meshList.at(i).F.cols(), meshList.at(i).F_vbo.scalar_type, 0);
			if(mode == 0)
			{
				// cout << "Wireframe" << endl;
				
				meshList.at(i).V_vbo.update(meshList.at(i).V);
				meshList.at(i).F_vbo.update(meshList.at(i).F);
				meshList.at(i).calculateColor();
				meshList.at(i).Color_vbo.update(meshList.at(i).color);
				meshList.at(i).Normal_vbo.update(meshList.at(i).normal);

				programGlobal.bindVertexAttribArray("position", meshList.at(i).V_vbo);
				programGlobal.bindVertexAttribArray("normal", meshList.at(i).Normal_vbo);
				programGlobal.bindVertexAttribArray("color", meshList.at(i).Color_vbo);

				drawWireframe(i);
			}
			else if(mode == 1)
			{
				// cout << "Phong" << endl;
				
				meshList.at(i).V_vbo.update(meshList.at(i).V);
				meshList.at(i).F_vbo.update(meshList.at(i).F);
				meshList.at(i).calculateColor();
				
				meshList.at(i).Normal_vbo.update(meshList.at(i).normal);
				meshList.at(i).Color_vbo.update(meshList.at(i).color);

				programGlobal.bindVertexAttribArray("position", meshList.at(i).V_vbo);
				programGlobal.bindVertexAttribArray("normal", meshList.at(i).Normal_vbo);
				programGlobal.bindVertexAttribArray("color", meshList.at(i).Color_vbo);

				glDrawElements(GL_TRIANGLES, 3 * meshList.at(i).F.cols(), meshList.at(i).F_vbo.scalar_type, 0);
			}
			if(mode == 2)
			{
				// cout << "Flat" << endl;
				meshList.at(i).V_vbo.update(meshList.at(i).V_flat);
				meshList.at(i).F_vbo.update(meshList.at(i).F_flat);
				meshList.at(i).calculateColor();
				
				// meshList.at(i).calculateNormalFlatVertex();
				meshList.at(i).Normal_vbo.update(meshList.at(i).normal_flat);
				meshList.at(i).Color_vbo.update(meshList.at(i).color);

				programGlobal.bindVertexAttribArray("position", meshList.at(i).V_vbo);
				programGlobal.bindVertexAttribArray("normal", meshList.at(i).Normal_vbo);
				programGlobal.bindVertexAttribArray("color", meshList.at(i).Color_vbo);

				glDrawElements(GL_TRIANGLES, 3 * meshList.at(i).F_flat.cols(), meshList.at(i).F_vbo.scalar_type, 0);

				Vector3f oldColor = meshList.at(i).currentColor;
				meshList.at(i).currentColor << 0,0,0;
				meshList.at(i).calculateColor();
				meshList.at(i).Color_vbo.update(meshList.at(i).color);
				programGlobal.bindVertexAttribArray("color", meshList.at(i).Color_vbo);
				
				drawWireframe(i);
				meshList.at(i).currentColor = oldColor;
				// drawWireframe(i);
			}
		}

		clickDrag();

		// Swap front and back buffers
		glfwSwapBuffers(window);

		// Poll for and process events
		glfwPollEvents();

		// exit(0);
	}

	// Deallocate opengl memory
	
	programGlobal.free();
	for(int i = 0; i < meshList.size(); i++)
	{
		Mesh bunny = meshList.at(i);
		bunny.vao.free();
		bunny.V_vbo.free();
		bunny.F_vbo.free();
	}


	// Deallocate glfw internals
	glfwTerminate();
	return 0;
}
