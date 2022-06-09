#include <iostream>
#include <SFML/Graphics.hpp>
#include <functional>
#include <cmath>
#include <map>
#include <set>
#include <fstream>
#include "SimplexNoise.h"

typedef uint8_t u8;
typedef uint32_t u32;
typedef int32_t s32;

typedef sf::Vector2u vec2u;
typedef sf::Vector2f vec2;
typedef sf::Vector3<u32> vec3u;
typedef sf::Vector3f vec3;

float Randf() {
	return (float)rand() / RAND_MAX;
}
vec3 RandUnit() {
	float y = Randf()*2-1;
	float phi = Randf()*M_PI*2;
	float r = sqrt(1-y*y);
	return {r*sinf(phi), y, r*cosf(phi)};
}
vec2 operator*(vec2 A, vec2 B) {
	return {A.x * B.x, A.y * B.y};
}
vec2 operator/(vec2 A, vec2 B) {
	return {A.x / B.x, A.y / B.y};
}
vec3 operator*(vec3 A, vec3 B) {
	return {A.x * B.x, A.y * B.y, A.z * B.z};
}
vec3 operator/(vec3 A, vec3 B) {
	return {A.x / B.x, A.y / B.y, A.z / B.z};
}
float dot(vec2 a, vec2 b) {
	return a.x*b.x+a.y*b.y;
}
float dot(vec3 a, vec3 b) {
	return a.x*b.x+a.y*b.y+a.z*b.z;
}
vec3 normalize(vec3 v, vec3 d = vec3(1,0,0), float eps = 1e-9) {
    float l = sqrt(dot(v, v));
    return l < eps ? d : v / l;
}
vec3 utof(vec3u v) {
	return vec3(v.x, v.y, v.z);
}
template<typename T>
float len2(T v) {
	return dot(v, v);
}
vec3 cross(vec3 a, vec3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

struct Trihedron {
	vec3 i, j, k;
	static Trihedron Make(vec3 u, vec3 v = vec3(1,0,0)) {
		vec3 i = normalize(u, vec3(1,0,0));
		vec3 j = normalize(v - i*dot(v,i), normalize(vec3(v.y, -v.x,0),vec3(1,0,0)));
		vec3 k = cross(i, j);
		return {i, j, k};
	}
	vec3 Descartes(vec3 p) {
		return i * p.x + j * p.y + k * p.z;
	}
	vec3 Cylindric(vec3 p) {
		return p.x * (j * cosf(p.y)  + k * sinf(p.y)) + i * p.z; 
	}
	vec3 Conic(vec3 p) {
		return Cylindric({p.z*tanf(p.x),p.y,p.z}); 		
	}
};

class Mesh3 {
	public:
	struct Triangle {
		u32 a, b, c;
	};
	private:
	std::vector<vec3> points;
	std::vector<Triangle> triangles;
	public:
	Mesh3(std::vector<vec3> points, std::vector<Triangle> triangles) {
		this->points = std::move(points);
		this->triangles = std::move(triangles);
	}
	Mesh3(std::map<u32, vec3> _points, std::vector<Triangle> _triangles) {
		points.reserve(_points.size());
		std::map<u32, u32> cids;
		u32 count = 0;
		for(const auto& P : _points) {
			cids[P.first] = count++;
			points.push_back(P.second); 
		}
		triangles = std::move(_triangles);
		for(Triangle& t : triangles) {
			t.a = cids[t.a];
			t.b = cids[t.b];
			t.c = cids[t.c];
		}
	}
	void WriteObj(std::ofstream& fout, const std::string& name, float scale = 5.f) {
		fout << "g " << name << "\n";
		for(vec3 p : points) {
			p *= scale;
			fout << "v " << p.x << " " << p.y << " " << p.z << "\n";		
		}

		for(const Triangle& t : triangles) {
			fout << "f " 
			<< (s32)(t.a-points.size()) << " " 
			<< (s32)(t.b-points.size()) << " "
			<< (s32)(t.c-points.size()) << "\n";
		}
	}
};
struct Mesh {
	struct Line {
		u32 a, b;
	};
	std::map<u32, vec2> points;
	std::vector<Line> lines;

	void DrawLine(sf::RenderTexture& rtex) {
		vec2 S = (vec2)rtex.getSize();
		u32 n = lines.size();
		sf::Vertex *vlines = new sf::Vertex[n * 2];
		for(u32 i = 0; i < n; i++) {
			vec2 pa = points[lines[i].a] * S, pb = points[lines[i].b] * S;
			vlines[i * 2] = pa;
			vlines[i * 2 + 1] = pb;
		} 

		rtex.draw(vlines, n * 2, sf::Lines);
		delete[] vlines;
	}
};
float Delta(float h0, float h1) {
	return h0 / (h0 - h1);
}
void AddEdges(bool* v, u32 vd[4], s32* e, u32 ed[4]) {
	for(u32 i = 0; i < 4; i++) {
		//printf("<8: %u %u>\n", vd[i], vd[(i+1)%4]);
		if(v[vd[i]]||!v[vd[(i+1)%4]]) continue;
		for(u32 j = (i+1)%4, k; j!=i; j = (j+1)%4) {
			//printf("<8: %u>\n", vd[(j+1)%4]);
			if(!v[vd[(j+1)%4]]) {
				//printf("<12: %u>\n", ed[i]);
				e[ed[i]] = ed[j];
				//printf("@");
				break;
			}
		}
	}
}
struct HeigthMap {
	vec2u size;
	std::vector<float> v;
	s32 Sh, Sv;
	HeigthMap(u32 W, u32 H, const std::function<float(vec2)>& f) : size(W, H), v((W + 1) * (H + 1)) {
		for(u32 y = 0, i = 0; y <= H; y++) {
			for(u32 x = 0; x <= W; x++, i++) {
				v[i] = f(vec2(((float)x/W)*2-1, ((float)y/H)*2-1));
			}
		}
		Sh = (size.y + 1) * size.x, Sv = (size.x + 1) * size.y;
	}
	float At(vec2u p) const {
		return v[(size.x+1)*p.y+p.x];
	}
	float At(u32 x, u32 y) const {
		return v[(size.x+1)*y+x];
	}
	sf::Image AsImage(float mi, float ma) const {
		sf::Image img;
		img.create(size.x + 1, size.y + 1);
		for(u32 y = 0, i = 0; y <= size.y; y++) {
			for(u32 x = 0; x <= size.x; x++, i++) {
				u32 h = (v[i] - mi)/(ma - mi) * 255;
				u8 h8 = h > 255 ? 255 : h < 0 ? 0 : h;
				img.setPixel(x, y, {h8,h8,h8,255});
			}
		}
		return img;
	}
	u32 EdgeId(s32 x, s32 y, s32 t) const {
		return t == 0 ? x+y*(s32)size.x : Sh+x+y*((s32)size.x+1);
	}
	Mesh GetMesh() const {
		std::vector<Mesh::Line> lines;
		std::set<u32> vids;
		std::map<u32, vec2> points;

		for(s32 y = 0; y < (s32)size.y; y++) {
			for(s32 x = 0; x < (s32)size.x; x++) {
				bool p[4] = {At(x, y)>0, At(x+1,y)>0, At(x+1,y+1)>0, At(x,y+1)>0};
				u32 ids[4] = {
					EdgeId(x,y,0), EdgeId(x+1,y,1),
					EdgeId(x,y+1,0), EdgeId(x,y,1)
				};
				s32 to[4] = {-1,-1,-1,-1};
				u32 td[4] = {0,1,2,3};
				AddEdges(p, td, to, td);
				for(u32 i = 0; i < 4; i++) {
					if(to[i]!=-1) {
						lines.push_back({(u32)ids[i], (u32)ids[to[i]]});
						vids.insert(ids[i]);
						vids.insert(ids[to[i]]);
					}
				}
			}
		}
		for(u32 id : vids) {
			if(id < Sh) {
				u32 x = id % size.x, y = id / size.x;
				points[id] = vec2(x + Delta(At(x, y), At(x + 1, y)), y) / (vec2)size;
			} else {
				u32 x = (id - Sh) % (size.x + 1), y = (id - Sh) / (size.x + 1);
				points[id] = vec2(x, y + Delta(At(x, y), At(x, y + 1))) / (vec2)size;
			}
		}
		return {points, lines};
	}
};
struct HeigthMap3 {
	vec3u size;
	std::vector<float> v;
	s32 Sx, Sy, Sz;

	HeigthMap3(u32 W, u32 H, u32 D, const std::function<float(vec3)>& f) : size(W, H, D), v((W + 1) * (H + 1) * (D+1)) {
		for(u32 z = 0, i = 0; z <= D; z++) {
			for(u32 y = 0; y <= H; y++) {
				for(u32 x = 0; x <= W; x++, i++) {
					v[i] = f(vec3(((float)x/W)*2-1, ((float)y/H)*2-1, ((float)z/D)*2-1));
				}
			}
		}
		Sx = size.x * (size.y + 1) * (size.z + 1);
		Sy = (size.x + 1) * size.y * (size.z + 1);
		Sz = (size.x + 1) * (size.y + 1) * size.z;
	}
	float At(vec3u p) const {
		return v[(size.x+1)*(size.y+1)*p.z+(size.x+1)*p.y+p.x];
	}
	float At(u32 x, u32 y, u32 z) const {
		return v[(size.x+1)*(size.y+1)*z+(size.x+1)*y+x];
	}
	u32 EdgeId(s32 x, s32 y, s32 z, s32 t) const {
		return t == 0 ? x+y*(s32)size.x+z*size.x*(size.y+1):
				t == 1 ? Sx+x+y*(s32)(size.x+1)+z*(size.x+1)*size.y:
				Sx+Sy+x+y*(s32)(size.x+1)+z*(size.x+1)*(size.y+1);
	}
	Mesh3 GetMesh(vec3 p0, float factor) const {
		s32 Sx = size.x * (size.y + 1) * (size.z + 1),
			Sy = (size.x + 1) * size.y * (size.z + 1),
			Sz = (size.x + 1) * (size.y + 1) * size.z;

	 	std::vector<Mesh3::Triangle> triangles;
	 	std::set<u32> vids;
	 	std::map<u32, vec3> points;

		for(s32 z = 0; z < (s32)size.z; z++) {
			for(s32 y = 0; y < (s32)size.y; y++) {
				for(s32 x = 0; x < (s32)size.x; x++) {
					bool p[8] = {At(x,y,z)>0, At(x+1,y,z)>0, At(x,y+1,z)>0, At(x+1,y+1,z)>0,
						At(x,y,z+1)>0, At(x+1,y,z+1)>0, At(x,y+1,z+1)>0, At(x+1,y+1,z+1)>0};
					s32 to[12] = {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};
					u32 ids[12] = {
						EdgeId(x,y,z,0),EdgeId(x,y+1,z,0),EdgeId(x,y,z+1,0),EdgeId(x,y+1,z+1,0),
						EdgeId(x,y,z,1),EdgeId(x+1,y,z,1),EdgeId(x,y,z+1,1),EdgeId(x+1,y,z+1,1),
						EdgeId(x,y,z,2),EdgeId(x+1,y,z,2),EdgeId(x,y+1,z,2),EdgeId(x+1,y+1,z,2)
					};
					u32 facesV[] = {
						0,1,3,2,
						4,6,7,5,
						0,4,5,1,
						2,3,7,6,
						0,2,6,4,
						1,5,7,3
					};
					u32 facesE[] = {
						0,5,1,4,
						6,3,7,2,
						8,2,9,0,
						1,11,3,10,
						4,10,6,8,
						9,7,11,5
					};
					for(u32 i = 0; i < 6; i++) {
						AddEdges(p, facesV + i * 4, to, facesE + i * 4);
					}
					for(s32 i = 0; i < 12; i++) {
						if(s32 j = to[i]; j!=-1) {
							for(; to[j] != i; j = to[j]) {
								if(to[j]==-1) {
									printf(" ~~~~ ");
									exit(1);
								}
								Mesh3::Triangle t = {ids[j], ids[i], ids[to[j]]};
								triangles.push_back(t);
								vids.insert(t.a);
								vids.insert(t.b);
								vids.insert(t.c);
							}

							for(j = i; j != -1;) {
								s32 k = to[j];
								to[j] = -1;
								j = k;
							}
						}						
					}
				}
			}
		}

		for(u32 id : vids) {
			if(id < Sx) {
				u32 x = id%size.x, y = (id/size.x)%(size.y+1), z = id/(size.x*(size.y+1));
				points[id] = (vec3(x + Delta(At(x, y, z), At(x + 1, y, z)), y, z)/utof(size) - vec3(.5,.5,.5))*factor;
			} else if (id < Sx + Sy) {
				id -= Sx;
				u32 x = id %(size.x+1), y = (id/(size.x+1))%size.y, z = id/((size.x+1)*size.y);
				points[id + Sx] = (vec3(x, y + Delta(At(x, y, z), At(x, y + 1, z)), z)/utof(size) - vec3(.5,.5,.5))*factor;
			} else {
				id -= Sx + Sy;				
				u32 x = id %(size.x+1), y = (id/(size.x+1))%(size.y+1), z = id/((size.x+1)*(size.y+1));
				points[id + Sx + Sy] = (vec3(x, y, z + Delta(At(x, y, z), At(x, y, z + 1)))/utof(size) - vec3(.5,.5,.5))*factor;
			}
		}	
	 	return {points, triangles};
	 }
};
Mesh3 PillarMesh(vec3 p0, vec3 dir, float hDown, float hTop, const std::vector<vec2>& vertices, u32 N) {
	Trihedron T = Trihedron::Make(dir);
	u32 L = vertices.size();
	std::vector<vec3> points(N*L + 2);
	std::vector<Mesh3::Triangle> triangles(2*(L-1)*N + N * 2);
	float phi = M_PI * 2.f / N;
	for(u32 l = 0, i = 0; l < L; l++) {
		for(u32 n = 0; n < N; n++, i++) {
			points[i] = T.Conic({vertices[l].x,phi*n,vertices[l].y});
		}
	}

	points[N*L] = T.Conic({0, 0, hTop});
	points[N*L+1] = T.Conic({0, 0, hDown});

	for(u32 l = 0, i = 0; l < L - 1; l++) {
		for(u32 n = 0; n < N; n++) {
			u32 v0 = n+l*N, v1 = (n+1)%N+l*N, v2 = n+(l+1)*N, v3 = (n+1)%N+(l+1)*N;
			triangles[i++] = {v0, v1, v3};
			triangles[i++] = {v0, v3, v2};
		}
	}
	for(u32 n = 0, i = 2*(L-1)*N; n < N; n++) {
		triangles[i++] = {n, N*L+1, (n+1)%N};
		triangles[i++] = {n+(L-1)*N, (n+1)%N+(L-1)*N, N*L};
	}
	return {points, triangles};
}
Mesh3 RingMesh(vec3 p0, vec3 dir0, float rIn, float rOut, float h, u32 N) {
	std::vector<vec3> points(4*N);
	std::vector<Mesh3::Triangle> triangles(N*8);
	Trihedron T = Trihedron::Make(dir0);

	float phi = M_PI*2/N;
	for(u32 n = 0; n < N; n++) {
		points[n*4] = T.Cylindric({rIn, phi*n, -h/2}) + p0;
		points[n*4+1] = T.Cylindric({rOut, phi*n, -h/2}) + p0;
		points[n*4+2] = T.Cylindric({rOut, phi*n, h/2}) + p0;
		points[n*4+3] = T.Cylindric({rIn, phi*n, h/2}) + p0;
	}
	for(u32 n = 0; n < N; n++) {
		for(u32 i = 0; i < 4; i++) {
			u32 v0 = 4*n+i, v1 = 4*n+(i+1)%4,
				v2 = (4*n+i+4)%(4*N), v3 = (4*n+(i+1)%4+4)%(4*N);

			triangles[8*n+i] = {v0, v2, v1};
			triangles[8*n+i+4] = {v2, v3, v1};
		}
	}
	return {points, triangles};
}
void WriteHeader(std::ofstream& fout, std::ofstream& foutmtl, const std::string& name) {
	fout.open(name+".obj");
	foutmtl.open(name+".mtl");
	fout << "mtlib " << name << ".mtl";
}
void WriteMaterialHeader(std::ofstream& fout, std::ofstream& foutmtl, const std::string& name, const std::string& mtl) {
	static u32 id = 0;
	fout << "usemtl mtl" << std::to_string(id) << "\no " << name << "\n";
	foutmtl << "newmtl mtl" << std::to_string(id) << "\n" << mtl << "\n";
	id++;
}
void DrawLine(sf::RenderTexture& rtex, vec2 A, vec2 B) {
	vec2 S = (vec2)rtex.getSize();
	sf::Vertex line[] = {
		S * A,
		S * B
	};
	rtex.draw(line, 2, sf::Lines);
}

#define DIM 2

int main() {
#if DIM == 2
	HeigthMap hm(100, 100, [] (vec2 v) {
		v *= 10.f;
		float x = v.x, y = v.y;
		return 4.f + .3*SimplexNoise::noise(x * .55, y * .55) - .05*len2(v);
	});

	sf::RenderTexture rtex;
	rtex.create(650, 650);
	rtex.clear({0,0,0,255});

	Mesh mesh = hm.GetMesh();
	mesh.DrawLine(rtex);
	rtex.getTexture().copyToImage().saveToFile("result.png");
#elif DIM == 3
	HeigthMap3 hm3(35, 35, 35, [] (vec3 v) {
		v *= 10.f;
		float x = v.x, y = v.y, z = v.z;
		return 4.f + .7*SimplexNoise::noise(x * .3, y * .3, z * .3) - .05*len2(v);
	});
	std::ofstream fout, foutmtl;
	WriteHeader(fout, foutmtl, "planet");
	WriteMaterialHeader(fout, foutmtl, "core", "Kd 1.0 1.0 1.0");
	hm3.GetMesh({0,0,0}, 20).WriteObj(fout, "core");
	WriteMaterialHeader(fout, foutmtl, "pillars", "Kd 0.2 0.2 0.2");
	for(u32 i = 0; i < 20; i++) {
		vec3 v = RandUnit();
		float h = Randf()*2.5f+8.5f;
		float r1 = .1f, r0 = .07f;
		float rf = Randf()+.5;
		PillarMesh(
			{0,0,0}, v,
			.0f, h+.9f,{
				{r1*rf, h},
				{r0*rf, h},
				{r0*rf, h+.3f},
				{r1*rf, h+.3f},
				{r1*rf, h+.6f},
			},
			4+rand()%10
		).WriteObj(fout, "pillar_" + std::to_string(i));
	}
	WriteMaterialHeader(fout, foutmtl, "rings", "Kd 0.4 0.4 0.4");
	float l0 = 12.3f, l = .8f, dl = .2f, h = .2f, ddl = .15f, phi = 0.f;
	for(u32 i = 0; i < 4; i++) {
		vec3 dir = {sinf(phi*i),cosf(phi*i),0};
		RingMesh(
			{0,0,0}, dir,
			l0, l0+l, h, 16
		).WriteObj(fout, "ring_0");		

		l0 += l + dl;
		l += ddl;
	}
#endif 
	return 0;
}