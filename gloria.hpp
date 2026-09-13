#pragma once

/* =====================================================================================================
   GLORIA - Grillas Locales de Refinamiento con Interpolacion Acustica
            (Grid-Local Overset Refinement with Interpolated Acoustics)

   Que es:
     Una capa de refinamiento de malla para FluidX3D que permite anidar grillas LBM a resolucion
     doble (dx/2, dt/2) alrededor de la geometria (el .stl), dentro de un dominio grueso grande.
     Cada nivel duplica la resolucion del anterior, asi que con L niveles la geometria se resuelve
     con celdas 2^L veces mas chicas que el dominio exterior, pagando solo el costo del parche.

   Como funciona (resumen tecnico, el tutorial TUTORIAL-GLORIA.md lo explica en detalle):
     - Cada nivel es una simulacion LBM completa de FluidX3D (su propia grilla, su propio contexto
       OpenCL, sus propios kernels). No se modifico NINGUN kernel del solver original.
     - Escalado acustico: dx_fino = dx/2 y dt_fino = dt/2. Con esta eleccion la velocidad de red
       (u_lbm) y la fluctuacion de densidad (rho-1) son IDENTICAS en todos los niveles: en las
       interfaces solo hay que interpolar, no re-escalar. Lo unico que cambia entre niveles es la
       viscosidad de red: nu_lbm se duplica por nivel (y tau se aleja de 0.5 -> el nivel fino es
       MAS estable que el grueso).
     - Acople grueso->fino (prolongacion): las capas exteriores del parche fino se marcan TYPE_E
       (frontera de equilibrio de FluidX3D). En cada paso grueso se leen del device unas "lonjas"
       del campo grueso alrededor del borde del parche, se interpola bilineal/trilineal en espacio
       y lineal en tiempo (el fino da 2 sub-pasos por paso grueso, con theta = 1/2 y 1), y se
       escriben rho y u en las celdas fantasma del device fino. El kernel les impone fi = feq.
     - Acople fino->grueso (restriccion): un anillo de celdas TYPE_E del nivel grueso, ubicado
       unas celdas adentro del parche, recibe el promedio 2x2(x2) de las celdas finas. Asi la
       solucion fina "manda" sobre la gruesa y la estela/las fuerzas se propagan hacia afuera.
     - La region gruesa en el interior profundo del parche sigue corriendo (con el solido
       voxelizado grueso) pero queda apantallada por el anillo: su solucion no se filtra afuera.

   Limitaciones conocidas (documentadas en el tutorial):
     - El acople por equilibrio descarta la parte de no-equilibrio (tensor viscoso) en la interfaz:
       error O(Ma^2, Kn). Regla practica: dejar las interfaces lejos de gradientes fuertes
       (>= 0.5 cuerdas del cuerpo, nunca cruzando la capa limite ni la estela cercana si se puede).
     - Extensiones no soportadas junto con GLORIA: SURFACE, TEMPERATURE, PARTICLES, multi-GPU por
       nivel (cada nivel usa 1 dominio; niveles distintos pueden ir en la misma GPU).
     - Geometria estatica (sin MOVING_BOUNDARIES a traves de la interfaz).

   Requiere: EQUILIBRIUM_BOUNDARIES en defines.hpp.
   ===================================================================================================== */

#include "lbm.hpp"
#include <vector>

#ifndef EQUILIBRIUM_BOUNDARIES
#error "GLORIA necesita #define EQUILIBRIUM_BOUNDARIES en defines.hpp (las interfaces usan celdas TYPE_E)."
#endif
#ifdef SURFACE
#error "GLORIA no soporta la extension SURFACE."
#endif
#ifdef TEMPERATURE
#error "GLORIA no soporta la extension TEMPERATURE."
#endif
#ifdef PARTICLES
#error "GLORIA no soporta la extension PARTICLES."
#endif

struct GloriaBox { // caja semiabierta [x0,x1) x [y0,y1) x [z0,z1) en celdas locales de una grilla
	uint x0=0u, x1=0u, y0=0u, y1=0u, z0=0u, z1=0u;
	GloriaBox() {}
	GloriaBox(const uint x0, const uint x1, const uint y0, const uint y1, const uint z0, const uint z1) : x0(x0), x1(x1), y0(y0), y1(y1), z0(z0), z1(z1) {}
};

struct GloriaGhost { // una celda fantasma del nivel fino, alimentada por interpolacion desde el padre
	ulong n = 0ull;     // indice lineal en la grilla fina
	ulong p[8] = {0ull,0ull,0ull,0ull,0ull,0ull,0ull,0ull}; // indices (host) del stencil en la grilla padre
	float w[8] = {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}; // pesos bilineales/trilineales
};

struct GloriaFeedback { // una celda del anillo del padre, alimentada por restriccion (promedio) del fino
	ulong n_parent = 0ull; // indice lineal en la grilla padre
	uint fx0=0u, fy0=0u, fz0=0u; // esquina del bloque fino 2x2(x2) que se promedia
};

class GloriaLBM {
public:
	// ---------------- parametros configurables (cambiar antes del primer run()) ----------------
	uint n_ghost   = 2u;   // espesor de la capa fantasma del nivel fino [celdas finas] (2 = recomendado)
	uint fb_margin = 3u;   // profundidad del anillo de retroalimentacion [celdas del padre] (>= n_ghost/2+1)
	bool two_way   = true; // acople bidireccional (false = el fino escucha pero no habla; solo para depurar)

private:
	struct Level {
		LBM* lbm = nullptr;
		Units u_si;                 // sistema de unidades SI de este nivel (dx y dt propios)
		uint scale = 1u;            // factor de refinamiento respecto del nivel 0 (= 2^nivel)
		float g0x=0.0f, g0y=0.0f, g0z=0.0f; // origen del recuadro de este nivel en coordenadas de celda del nivel 0
		// --- acople con el padre (solo niveles >= 1) ---
		int px0=0, py0=0, pz0=0;    // origen del parche en celdas locales del padre
		uint sx=0u, sy=0u, sz=0u;   // tamano del parche en celdas del padre
		std::vector<GloriaGhost> ghosts;        // celdas fantasma del fino
		std::vector<float> ghost_old, ghost_new; // (rho,ux,uy,uz) por celda fantasma, en t y t+1 del padre
		std::vector<GloriaBox> ghost_boxes;       // sub-cajas finas que se escriben al device (capas fantasma)
		std::vector<GloriaBox> parent_read_boxes; // lonjas del padre que se leen del device (soporte de interpolacion)
		std::vector<GloriaFeedback> feedback;     // celdas del anillo del padre
		std::vector<GloriaBox> fine_read_boxes;   // sub-cajas finas que se leen para la restriccion
		std::vector<GloriaBox> parent_write_boxes;// lineas/caras del anillo del padre que se escriben al device
	};
	std::vector<Level> levels;
	bool is2D = false;        // Nz==1 en el nivel 0 (modo D2Q9)
	bool initialized_ = false;
	bool units_set = false;
	float si_rho_ = 1.0f;

	static ulong idx3(const uint x, const uint y, const uint z, const uint Nx, const uint Ny) {
		return (ulong)x+((ulong)y+(ulong)z*(ulong)Ny)*(ulong)Nx;
	}

	// lee del device las lonjas (rho, u) de la grilla 'lbm' indicadas por 'boxes'; los datos quedan en los arrays host en los mismos indices
	void read_boxes(LBM* lbm, const std::vector<GloriaBox>& boxes) {
#ifndef UPDATE_FIELDS
		lbm->update_fields(); // asegurar que rho/u en device esten al dia (con INTERACTIVE_GRAPHICS ya lo estan en cada paso)
#endif // UPDATE_FIELDS
		const uint Nx=lbm->get_Nx(), Ny=lbm->get_Ny(), Nz=lbm->get_Nz();
		for(const GloriaBox& b : boxes) {
			lbm->lbm_domain[0]->rho.read_from_device_3d(b.x0, b.x1, b.y0, b.y1, b.z0, b.z1, Nx, Ny, Nz);
			lbm->lbm_domain[0]->u  .read_from_device_3d(b.x0, b.x1, b.y0, b.y1, b.z0, b.z1, Nx, Ny, Nz);
		}
	}
	void write_boxes(LBM* lbm, const std::vector<GloriaBox>& boxes) {
		const uint Nx=lbm->get_Nx(), Ny=lbm->get_Ny(), Nz=lbm->get_Nz();
		for(const GloriaBox& b : boxes) {
			lbm->lbm_domain[0]->rho.write_to_device_3d(b.x0, b.x1, b.y0, b.y1, b.z0, b.z1, Nx, Ny, Nz);
			lbm->lbm_domain[0]->u  .write_to_device_3d(b.x0, b.x1, b.y0, b.y1, b.z0, b.z1, Nx, Ny, Nz);
		}
	}

	// evalua el stencil de interpolacion de todas las celdas fantasma del nivel l usando los arrays HOST del padre, y lo guarda en 'target'
	void ghosts_from_parent_host(const uint l, std::vector<float>& target) {
		Level& C = levels[l];
		LBM* plbm = levels[l-1u].lbm;
		Memory<float>& prho = plbm->lbm_domain[0]->rho;
		Memory<float>& pu   = plbm->lbm_domain[0]->u;
		const ulong pN = prho.length();
		const ulong G = (ulong)C.ghosts.size();
		target.resize(G*4ull);
		parallel_for(G, [&](ulong i) {
			const GloriaGhost& g = C.ghosts[i];
			float r=0.0f, ux=0.0f, uy=0.0f, uz=0.0f;
			for(uint k=0u; k<8u; k++) {
				const float wk = g.w[k];
				if(wk!=0.0f) {
					const ulong p = g.p[k];
					r  += wk*prho[p];
					ux += wk*pu[p];
					uy += wk*pu[pN+p];
					uz += wk*pu[2ull*pN+p];
				}
			}
			target[4ull*i   ] = r;
			target[4ull*i+1u] = ux;
			target[4ull*i+2u] = uy;
			target[4ull*i+3u] = uz;
		});
	}

	// escribe en el device fino las celdas fantasma con la mezcla temporal (1-theta)*viejo + theta*nuevo
	void write_ghosts(const uint l, const float theta) {
		Level& C = levels[l];
		Memory<float>& frho = C.lbm->lbm_domain[0]->rho;
		Memory<float>& fu   = C.lbm->lbm_domain[0]->u;
		const ulong fN = frho.length();
		const ulong G = (ulong)C.ghosts.size();
		parallel_for(G, [&](ulong i) {
			const ulong n = C.ghosts[i].n;
			frho[n]          = (1.0f-theta)*C.ghost_old[4ull*i   ]+theta*C.ghost_new[4ull*i   ];
			fu[n]            = (1.0f-theta)*C.ghost_old[4ull*i+1u]+theta*C.ghost_new[4ull*i+1u];
			fu[fN+n]         = (1.0f-theta)*C.ghost_old[4ull*i+2u]+theta*C.ghost_new[4ull*i+2u];
			fu[2ull*fN+n]    = (1.0f-theta)*C.ghost_old[4ull*i+3u]+theta*C.ghost_new[4ull*i+3u];
		});
		write_boxes(C.lbm, C.ghost_boxes);
	}

	// restriccion fino->padre: promedio 2x2(x2) de rho/u del fino sobre las celdas del anillo del padre, y escritura al device del padre
	void feedback_to_parent(const uint l) {
		Level& C = levels[l];
		LBM* flbm = C.lbm;
		LBM* plbm = levels[l-1u].lbm;
		read_boxes(flbm, C.fine_read_boxes);
		Memory<float>& frho = flbm->lbm_domain[0]->rho;
		Memory<float>& fu   = flbm->lbm_domain[0]->u;
		Memory<uchar>& ffl  = flbm->lbm_domain[0]->flags;
		Memory<float>& prho = plbm->lbm_domain[0]->rho;
		Memory<float>& pu   = plbm->lbm_domain[0]->u;
		const ulong fN = frho.length(), pN = prho.length();
		const uint fNx=flbm->get_Nx(), fNy=flbm->get_Ny();
		const uint cz = is2D ? 1u : 2u; // celdas finas promediadas en z
		const ulong F = (ulong)C.feedback.size();
		parallel_for(F, [&](ulong i) {
			const GloriaFeedback& f = C.feedback[i];
			float r=0.0f, ux=0.0f, uy=0.0f, uz=0.0f; uint cnt=0u;
			for(uint dz=0u; dz<cz; dz++) for(uint dy=0u; dy<2u; dy++) for(uint dx=0u; dx<2u; dx++) {
				const ulong n = idx3(f.fx0+dx, f.fy0+dy, f.fz0+dz, fNx, fNy);
				if(!(ffl[n]&TYPE_S)) {
					r += frho[n]; ux += fu[n]; uy += fu[fN+n]; uz += fu[2ull*fN+n]; cnt++;
				}
			}
			if(cnt>0u) {
				const float inv = 1.0f/(float)cnt;
				prho[f.n_parent]          = r*inv;
				pu[f.n_parent]            = ux*inv;
				pu[pN+f.n_parent]         = uy*inv;
				pu[2ull*pN+f.n_parent]    = uz*inv;
			}
		});
		write_boxes(plbm, C.parent_write_boxes);
	}

	// un paso de tiempo del nivel l (recursivo: el hijo da 2 sub-pasos con fantasmas interpolados en tiempo, y despues retroalimenta)
	void advance(const uint l) {
		levels[l].lbm->run(1u);
		if(!running) return;
		if(l+1u<(uint)levels.size()) {
			const uint c = l+1u;
			Level& C = levels[c];
			read_boxes(levels[l].lbm, C.parent_read_boxes); // lonjas del padre en t+dt (recien dado el paso)
			ghosts_from_parent_host(c, C.ghost_new);
			for(uint sub=0u; sub<2u; sub++) {
				write_ghosts(c, 0.5f*(float)(sub+1u)); // theta = 1/2 y 1: interpolacion temporal entre t y t+dt del padre
				advance(c);
				if(!running) return;
			}
			C.ghost_old.swap(C.ghost_new);
			if(two_way) feedback_to_parent(c);
		}
	}

	// arma listas de celdas fantasma/anillo, marca los flags TYPE_E de interfaz y verifica la configuracion
	void finalize_interfaces() {
		for(uint l=1u; l<(uint)levels.size(); l++) {
			Level& C = levels[l];
			Level& P = levels[l-1u];
			LBM* flbm = C.lbm;
			LBM* plbm = P.lbm;
			if(flbm->get_D()!=1u||plbm->get_D()!=1u) print_error("GLORIA: cada nivel debe usar 1 solo dominio (multi-GPU por nivel no soportado).");
			const uint fNx=flbm->get_Nx(), fNy=flbm->get_Ny(), fNz=flbm->get_Nz();
			const uint pNx=plbm->get_Nx(), pNy=plbm->get_Ny(), pNz=plbm->get_Nz();
			const uint ng = n_ghost, m = fb_margin;
			if(m<ng/2u+1u) print_error("GLORIA: fb_margin ("+to_string(m)+") debe ser >= n_ghost/2+1 ("+to_string(ng/2u+1u)+").");
			if(C.sx<=2u*m+2u||C.sy<=2u*m+2u||(!is2D&&C.sz<=2u*m+2u)) print_error("GLORIA: el parche del nivel "+to_string(l)+" es demasiado chico para fb_margin="+to_string(m)+".");

			// ---- 1. celdas fantasma del fino: capas exteriores de espesor n_ghost -> TYPE_E + stencil de interpolacion ----
			const uint di = ng/2u+2u; // profundidad de la lonja del padre hacia adentro del parche
			const uint dovr = 2u;     // profundidad hacia afuera
			for(uint z=0u; z<fNz; z++) for(uint y=0u; y<fNy; y++) for(uint x=0u; x<fNx; x++) {
				const bool gx = x<ng||x>=fNx-ng;
				const bool gy = y<ng||y>=fNy-ng;
				const bool gz = !is2D&&(z<ng||z>=fNz-ng);
				if(!(gx||gy||gz)) continue;
				const ulong n = idx3(x, y, z, fNx, fNy);
				if(flbm->flags[n]&TYPE_S) print_error("GLORIA: el solido toca el borde del parche del nivel "+to_string(l)+" en ("+to_string(x)+","+to_string(y)+","+to_string(z)+"). Agranda el parche.");
				flbm->flags[n] = TYPE_E;
				GloriaGhost g;
				g.n = n;
				// coordenada continua de la celda fina en el sistema local del padre (centros de celda en entero+0.5)
				const float qx = (float)C.px0+((float)x+0.5f)*0.5f;
				const float qy = (float)C.py0+((float)y+0.5f)*0.5f;
				const float qz = is2D ? 0.5f : (float)C.pz0+((float)z+0.5f)*0.5f;
				const int bx = (int)floorf(qx-0.5f), by = (int)floorf(qy-0.5f);
				const int bz = is2D ? 0 : (int)floorf(qz-0.5f);
				const float wx = qx-0.5f-(float)bx, wy = qy-0.5f-(float)by;
				const float wz = is2D ? 0.0f : qz-0.5f-(float)bz;
				const uint kz = is2D ? 1u : 2u;
				uint k = 0u;
				for(uint dz=0u; dz<kz; dz++) for(uint dy=0u; dy<2u; dy++) for(uint dx=0u; dx<2u; dx++) {
					const int X=bx+(int)dx, Y=by+(int)dy, Z=bz+(int)dz;
					if(X<0||Y<0||Z<0||X>=(int)pNx||Y>=(int)pNy||Z>=(int)pNz) print_error("GLORIA: el stencil de interpolacion del nivel "+to_string(l)+" se sale del padre. El parche debe estar al menos a 2 celdas del borde del dominio padre.");
					g.p[k] = idx3((uint)X, (uint)Y, (uint)Z, pNx, pNy);
					g.w[k] = (dx?wx:1.0f-wx)*(dy?wy:1.0f-wy)*(is2D?1.0f:(dz?wz:1.0f-wz));
					k++;
				}
				C.ghosts.push_back(g);
			}
			C.ghost_old.resize(C.ghosts.size()*4ull);
			C.ghost_new.resize(C.ghosts.size()*4ull);

			// sub-cajas finas para subir los fantasmas al device (los solapes en las esquinas son inofensivos)
			C.ghost_boxes.clear();
			C.ghost_boxes.push_back(GloriaBox(0u, ng, 0u, fNy, 0u, fNz));
			C.ghost_boxes.push_back(GloriaBox(fNx-ng, fNx, 0u, fNy, 0u, fNz));
			C.ghost_boxes.push_back(GloriaBox(0u, fNx, 0u, ng, 0u, fNz));
			C.ghost_boxes.push_back(GloriaBox(0u, fNx, fNy-ng, fNy, 0u, fNz));
			if(!is2D) {
				C.ghost_boxes.push_back(GloriaBox(0u, fNx, 0u, fNy, 0u, ng));
				C.ghost_boxes.push_back(GloriaBox(0u, fNx, 0u, fNy, fNz-ng, fNz));
			}

			// lonjas del padre que cubren el soporte de interpolacion de cada cara
			const int x0=C.px0, y0=C.py0, z0=C.pz0;
			const int x1=C.px0+(int)C.sx, y1=C.py0+(int)C.sy, z1=C.pz0+(int)C.sz;
			auto clampbox = [&](int a0, int a1, int b0, int b1, int c0, int c1) {
				if(a0<0||b0<0||c0<0||a1>(int)pNx||b1>(int)pNy||c1>(int)pNz) print_error("GLORIA: la lonja de lectura del padre se sale del dominio. Aleja el parche del borde (>= 2 celdas del padre).");
				return GloriaBox((uint)a0, (uint)a1, (uint)b0, (uint)b1, (uint)c0, (uint)c1);
			};
			C.parent_read_boxes.clear();
			const int zr0 = is2D ? 0 : z0-(int)dovr, zr1 = is2D ? 1 : z1+(int)dovr;
			C.parent_read_boxes.push_back(clampbox(x0-(int)dovr, x0+(int)di, y0-(int)dovr, y1+(int)dovr, zr0, zr1)); // cara x-
			C.parent_read_boxes.push_back(clampbox(x1-(int)di, x1+(int)dovr, y0-(int)dovr, y1+(int)dovr, zr0, zr1)); // cara x+
			C.parent_read_boxes.push_back(clampbox(x0-(int)dovr, x1+(int)dovr, y0-(int)dovr, y0+(int)di, zr0, zr1)); // cara y-
			C.parent_read_boxes.push_back(clampbox(x0-(int)dovr, x1+(int)dovr, y1-(int)di, y1+(int)dovr, zr0, zr1)); // cara y+
			if(!is2D) {
				C.parent_read_boxes.push_back(clampbox(x0-(int)dovr, x1+(int)dovr, y0-(int)dovr, y1+(int)dovr, z0-(int)dovr, z0+(int)di)); // cara z-
				C.parent_read_boxes.push_back(clampbox(x0-(int)dovr, x1+(int)dovr, y0-(int)dovr, y1+(int)dovr, z1-(int)di, z1+(int)dovr)); // cara z+
			}

			// ---- 2. anillo de retroalimentacion del padre: cascara del recuadro [x0+m, x1-m) -> TYPE_E + lista de restriccion ----
			C.feedback.clear();
			const int rx0=x0+(int)m, rx1=x1-(int)m, ry0=y0+(int)m, ry1=y1-(int)m;
			const int rz0 = is2D ? 0 : z0+(int)m, rz1 = is2D ? 1 : z1-(int)m;
			for(int Z=rz0; Z<rz1; Z++) for(int Y=ry0; Y<ry1; Y++) for(int X=rx0; X<rx1; X++) {
				const bool sx_ = X==rx0||X==rx1-1;
				const bool sy_ = Y==ry0||Y==ry1-1;
				const bool sz_ = !is2D&&(Z==rz0||Z==rz1-1);
				if(!(sx_||sy_||sz_)) continue;
				const ulong np = idx3((uint)X, (uint)Y, (uint)Z, pNx, pNy);
				if(plbm->flags[np]&TYPE_S) print_error("GLORIA: el solido (voxelizado grueso) toca el anillo de retroalimentacion del nivel "+to_string(l-1u)+". Agranda el parche o subi fb_margin.");
				plbm->flags[np] = TYPE_E;
				GloriaFeedback f;
				f.n_parent = np;
				f.fx0 = (uint)((X-x0)*2);
				f.fy0 = (uint)((Y-y0)*2);
				f.fz0 = is2D ? 0u : (uint)((Z-z0)*2);
				C.feedback.push_back(f);
			}

			// sub-cajas finas a leer para la restriccion (caras del anillo escaladas x2, espesor 2 celdas finas)
			C.fine_read_boxes.clear();
			const uint a0=(uint)((rx0-x0)*2), a1=(uint)((rx1-x0)*2), b0=(uint)((ry0-y0)*2), b1=(uint)((ry1-y0)*2);
			const uint c0 = is2D ? 0u : (uint)((rz0-z0)*2), c1 = is2D ? 1u : (uint)((rz1-z0)*2);
			const uint fz0 = is2D ? 0u : c0, fz1 = is2D ? 1u : c1;
			C.fine_read_boxes.push_back(GloriaBox(a0, a0+2u, b0, b1, fz0, fz1));
			C.fine_read_boxes.push_back(GloriaBox(a1-2u, a1, b0, b1, fz0, fz1));
			C.fine_read_boxes.push_back(GloriaBox(a0, a1, b0, b0+2u, fz0, fz1));
			C.fine_read_boxes.push_back(GloriaBox(a0, a1, b1-2u, b1, fz0, fz1));
			if(!is2D) {
				C.fine_read_boxes.push_back(GloriaBox(a0, a1, b0, b1, c0, c0+2u));
				C.fine_read_boxes.push_back(GloriaBox(a0, a1, b0, b1, c1-2u, c1));
			}

			// lineas/caras del anillo del padre a escribir al device (espesor 1 celda del padre)
			C.parent_write_boxes.clear();
			const uint wz0 = is2D ? 0u : (uint)rz0, wz1 = is2D ? 1u : (uint)rz1;
			C.parent_write_boxes.push_back(GloriaBox((uint)rx0, (uint)rx0+1u, (uint)ry0, (uint)ry1, wz0, wz1));
			C.parent_write_boxes.push_back(GloriaBox((uint)rx1-1u, (uint)rx1, (uint)ry0, (uint)ry1, wz0, wz1));
			C.parent_write_boxes.push_back(GloriaBox((uint)rx0, (uint)rx1, (uint)ry0, (uint)ry0+1u, wz0, wz1));
			C.parent_write_boxes.push_back(GloriaBox((uint)rx0, (uint)rx1, (uint)ry1-1u, (uint)ry1, wz0, wz1));
			if(!is2D) {
				C.parent_write_boxes.push_back(GloriaBox((uint)rx0, (uint)rx1, (uint)ry0, (uint)ry1, wz0, wz0+1u));
				C.parent_write_boxes.push_back(GloriaBox((uint)rx0, (uint)rx1, (uint)ry0, (uint)ry1, wz1-1u, wz1));
			}
			print_info("GLORIA: nivel "+to_string(l)+" acoplado: "+to_string((uint)C.ghosts.size())+" celdas fantasma, "+to_string((uint)C.feedback.size())+" celdas de anillo.");
		}
	}

public:
	GloriaLBM(const uint3 N, const float nu, const float fx=0.0f, const float fy=0.0f, const float fz=0.0f) {
		Level L;
		L.lbm = new LBM(N, nu, fx, fy, fz);
		L.scale = 1u;
		levels.push_back(L);
		is2D = N.z==1u;
	}
	~GloriaLBM() {
		for(int l=(int)levels.size()-1; l>=0; l--) delete levels[l].lbm;
	}
	GloriaLBM(const GloriaLBM&) = delete;
	GloriaLBM& operator=(const GloriaLBM&) = delete;

	// agrega un nivel de refinamiento (resolucion x2 respecto del ultimo nivel agregado) cubriendo el recuadro
	// [gx0,gx1) x [gy0,gy1) (x [gz0,gz1) en 3D), dado en COORDENADAS DE CELDA DEL NIVEL 0 (como lbm.center() etc).
	// El recuadro se redondea a celdas enteras del nivel padre.
	void add_level(const float gx0, const float gy0, const float gx1, const float gy1, const float gz0=0.0f, const float gz1=1.0f) {
		if(initialized_) print_error("GLORIA: add_level() debe llamarse antes del primer run().");
		Level& P = levels.back();
		const uint ps = P.scale;
		Level C;
		C.scale = 2u*ps;
		// redondear a celdas enteras del padre (en coords locales del padre)
		C.px0 = (int)roundf((gx0-P.g0x)*(float)ps); C.py0 = (int)roundf((gy0-P.g0y)*(float)ps);
		const int px1 = (int)roundf((gx1-P.g0x)*(float)ps), py1 = (int)roundf((gy1-P.g0y)*(float)ps);
		C.sx = (uint)max(0, px1-C.px0); C.sy = (uint)max(0, py1-C.py0);
		if(is2D) { C.pz0 = 0; C.sz = 1u; } else {
			C.pz0 = (int)roundf((gz0-P.g0z)*(float)ps);
			const int pz1 = (int)roundf((gz1-P.g0z)*(float)ps);
			C.sz = (uint)max(0, pz1-C.pz0);
		}
		const int margin = 2; // el parche debe estar a >= 2 celdas del borde del dominio padre (soporte de interpolacion)
		if(C.px0<margin||C.py0<margin||(int)C.sx<=0||(int)C.sy<=0||
		   C.px0+(int)C.sx>(int)P.lbm->get_Nx()-margin||C.py0+(int)C.sy>(int)P.lbm->get_Ny()-margin||
		   (!is2D&&(C.pz0<margin||(int)C.sz<=0||C.pz0+(int)C.sz>(int)P.lbm->get_Nz()-margin)))
			print_error("GLORIA: el recuadro del nivel "+to_string((uint)levels.size())+" se sale del padre o esta a menos de 2 celdas de su borde.");
		C.g0x = P.g0x+(float)C.px0/(float)ps;
		C.g0y = P.g0y+(float)C.py0/(float)ps;
		C.g0z = is2D ? 0.0f : P.g0z+(float)C.pz0/(float)ps;
		const uint fNx=2u*C.sx, fNy=2u*C.sy, fNz=is2D?1u:2u*C.sz;
		const float nu_f = 2.0f*P.lbm->get_nu(); // escalado acustico: nu_lbm se duplica por nivel
		const float fx_f = 0.5f*P.lbm->get_fx(), fy_f = 0.5f*P.lbm->get_fy(), fz_f = 0.5f*P.lbm->get_fz(); // fuerza por volumen: f_lbm se divide por 2
		C.lbm = new LBM(uint3(fNx, fNy, fNz), nu_f, fx_f, fy_f, fz_f);
		levels.push_back(C);
		print_info("GLORIA: nivel "+to_string((uint)levels.size()-1u)+" creado: "+to_string(fNx)+"x"+to_string(fNy)+"x"+to_string(fNz)+" celdas (dx = dx0/"+to_string(C.scale)+").");
	}

	uint get_levels() const { return (uint)levels.size(); }
	LBM& lbm(const uint l) { return *levels[l].lbm; }
	LBM& coarse() { return *levels.front().lbm; }
	LBM& finest() { return *levels.back().lbm; }
	Units& units_of(const uint l) { return levels[l].u_si; }
	uint scale_of(const uint l) const { return levels[l].scale; }
	ulong get_t() const { return levels[0].lbm->get_t(); } // pasos de tiempo del NIVEL 0

	// fija las unidades SI igual que units.set_m_kg_s(...), con x medido en CELDAS DEL NIVEL 0.
	// Deriva automaticamente las unidades de todos los niveles (dx y dt se dividen por 2 por nivel; u_lbm y rho son iguales en todos).
	void set_si_units(const float x, const float u, const float rho, const float si_x, const float si_u, const float si_rho) {
		const float unit_m0 = si_x/x;
		const float unit_s0 = u/si_u*unit_m0;
		si_rho_ = si_rho/rho;
		for(uint l=0u; l<(uint)levels.size(); l++) {
			const float unit_m = unit_m0/(float)levels[l].scale;
			const float unit_s = unit_s0/(float)levels[l].scale;
			levels[l].u_si.set_m_kg_s(unit_m, si_rho_*cb(unit_m), unit_s);
		}
		units.set_m_kg_s(unit_m0, si_rho_*cb(unit_m0), unit_s0); // el objeto global 'units' queda en las unidades del nivel 0
		units_set = true;
	}

	// voxeliza el .stl en TODOS los niveles. 'center' y 'size' en coordenadas/celdas del NIVEL 0 (mismas convenciones que LBM::voxelize_stl).
	void voxelize_stl(const string& path, const float3& center, const float3x3& rotation, const float size, const uchar flag=TYPE_S) {
		if(initialized_) print_error("GLORIA: voxelize_stl() debe llamarse antes del primer run().");
		for(uint l=0u; l<(uint)levels.size(); l++) {
			Level& L = levels[l];
			const float s = (float)L.scale;
			float3 c = float3((center.x-L.g0x)*s, (center.y-L.g0y)*s, is2D ? L.lbm->center().z : (center.z-L.g0z)*s);
			L.lbm->voxelize_stl(path, c, rotation, size*s, flag);
		}
	}
	void voxelize_stl(const string& path, const float3& center, const float size, const uchar flag=TYPE_S) {
		voxelize_stl(path, center, float3x3(1.0f), size, flag);
	}

	// ejecuta 'f(lbm, nivel)' sobre cada nivel: comodo para condiciones iniciales (velocidad de red identica en todos los niveles)
	template<class F> void for_each_level(F f) {
		for(uint l=0u; l<(uint)levels.size(); l++) f(*levels[l].lbm, l);
	}

	// convierte una posicion en celdas del nivel 0 a celdas locales del nivel l (para marcar flags, sondas, etc.)
	float3 to_level(const uint l, const float3& p0) const {
		const Level& L = levels[l];
		const float s = (float)L.scale;
		return float3((p0.x-L.g0x)*s, (p0.y-L.g0y)*s, is2D ? 0.5f : (p0.z-L.g0z)*s);
	}

	// corre 'steps' pasos del NIVEL 0 (cada uno arrastra 2 sub-pasos del nivel 1, 4 del nivel 2, ...).
	// La primera llamada configura las interfaces e inicializa todos los niveles; run(0) solo inicializa.
	void run(const ulong steps=max_ulong) {
		if(!initialized_) {
			if(!units_set) print_warning("GLORIA: no llamaste set_si_units(); los .vtk y las fuerzas SI van a salir en unidades de red.");
			finalize_interfaces();
			for(uint l=0u; l<(uint)levels.size(); l++) levels[l].lbm->run(0ull); // inicializa (sube flags/rho/u, calcula feq)
			for(uint l=1u; l<(uint)levels.size(); l++) ghosts_from_parent_host(l, levels[l].ghost_old); // fantasmas iniciales desde la condicion inicial del padre
			initialized_ = true;
			print_info("GLORIA: "+to_string((uint)levels.size())+" niveles inicializados.");
		}
		for(ulong i=0ull; i<steps&&running; i++) advance(0u);
	}

#ifdef GRAPHICS
	void show_level(const uint l) { info.lbm = levels[l].lbm; } // elige que nivel muestra la ventana interactiva
#endif // GRAPHICS

#ifdef FORCE_FIELD
	// fuerza/momento del fluido sobre las celdas 'flag_marker' del nivel l, en unidades de red DE ESE NIVEL
	float3 object_force(const uint l, const uchar flag_marker=TYPE_S) { return levels[l].lbm->object_force(flag_marker); }
	float3 object_torque(const uint l, const float3& rotation_center, const uchar flag_marker=TYPE_S) { return levels[l].lbm->object_torque(rotation_center, flag_marker); }
	// lo mismo pero convertido a SI con las unidades del nivel correspondiente
	float3 si_object_force(const uint l, const uchar flag_marker=TYPE_S) {
		const float3 F = object_force(l, flag_marker);
		return float3(levels[l].u_si.si_F(F.x), levels[l].u_si.si_F(F.y), levels[l].u_si.si_F(F.z));
	}
	float3 si_object_torque(const uint l, const float3& rotation_center, const uchar flag_marker=TYPE_S) {
		const float3 M = object_torque(l, rotation_center, flag_marker);
		return float3(levels[l].u_si.si_M(M.x), levels[l].u_si.si_M(M.y), levels[l].u_si.si_M(M.z));
	}
#endif // FORCE_FIELD

	// diagnostico: error RMS de velocidad (en unidades de red) entre el campo fino restringido (promedio 2x2(x2))
	// y el campo libre del padre, medido sobre la cascara de celdas del padre a profundidad n_ghost/2+1 adentro del
	// parche (la "franja" entre la capa fantasma y el anillo). Dividido por u_lbm de la corriente libre deberia dar
	// del orden de 1% o menos en flujo suave: es la forma rapida de validar que el acople esta funcionando.
	float interface_mismatch(const uint l) {
		if(l<1u||l>=(uint)levels.size()||!initialized_) return -1.0f;
		Level& C = levels[l];
		LBM* flbm = C.lbm;
		LBM* plbm = levels[l-1u].lbm;
		const uint fNx=flbm->get_Nx(), fNy=flbm->get_Ny();
		const uint fNz=flbm->get_Nz();
		const uint d = n_ghost/2u+1u; // profundidad de la banda de control [celdas del padre]
		if(d>=fb_margin) return -1.0f; // la banda debe caer antes del anillo
		// cascara del recuadro [x0+d, x1-d) del padre (en coords locales del padre)
		const int x0=C.px0, y0=C.py0, z0=C.pz0;
		const int x1=C.px0+(int)C.sx, y1=C.py0+(int)C.sy, z1=C.pz0+(int)C.sz;
		const int bx0=x0+(int)d, bx1=x1-(int)d, by0=y0+(int)d, by1=y1-(int)d;
		const int bz0 = is2D ? 0 : z0+(int)d, bz1 = is2D ? 1 : z1-(int)d;
		// leer del device las sub-cajas finas que mapean a la cascara de control
		std::vector<GloriaBox> fboxes;
		const uint a0=(uint)((bx0-x0)*2), a1=(uint)((bx1-x0)*2), b0=(uint)((by0-y0)*2), b1=(uint)((by1-y0)*2);
		const uint c0 = is2D ? 0u : (uint)((bz0-z0)*2), c1 = is2D ? 1u : (uint)((bz1-z0)*2);
		fboxes.push_back(GloriaBox(a0, a0+2u, b0, b1, c0, c1));
		fboxes.push_back(GloriaBox(a1-2u, a1, b0, b1, c0, c1));
		fboxes.push_back(GloriaBox(a0, a1, b0, b0+2u, c0, c1));
		fboxes.push_back(GloriaBox(a0, a1, b1-2u, b1, c0, c1));
		if(!is2D) {
			fboxes.push_back(GloriaBox(a0, a1, b0, b1, c0, c0+2u));
			fboxes.push_back(GloriaBox(a0, a1, b0, b1, c1-2u, c1));
		}
		read_boxes(flbm, fboxes);
		read_boxes(plbm, C.parent_read_boxes); // las lonjas del padre cubren la banda de control (di = n_ghost/2+2 > d)
		Memory<float>& fu = flbm->lbm_domain[0]->u;
		Memory<uchar>& ffl = flbm->lbm_domain[0]->flags;
		Memory<float>& pu = plbm->lbm_domain[0]->u;
		Memory<uchar>& pfl = plbm->lbm_domain[0]->flags;
		const ulong fN = (ulong)fNx*(ulong)fNy*(ulong)fNz;
		const ulong pN = pu.length(); // length() devuelve N (celdas), no N*dimensiones
		const uint pNx=plbm->get_Nx(), pNy=plbm->get_Ny();
		const uint cz = is2D ? 1u : 2u;
		double sum = 0.0; ulong cnt = 0ull;
		for(int Z=bz0; Z<bz1; Z++) for(int Y=by0; Y<by1; Y++) for(int X=bx0; X<bx1; X++) {
			const bool onx = X==bx0||X==bx1-1, ony = Y==by0||Y==by1-1;
			const bool onz = !is2D&&(Z==bz0||Z==bz1-1);
			if(!(onx||ony||onz)) continue;
			const ulong np = idx3((uint)X, (uint)Y, (uint)Z, pNx, pNy);
			if(pfl[np]&(TYPE_S|TYPE_E)) continue; // solo celdas libres del padre
			float ux=0.0f, uy=0.0f, uz=0.0f; uint c=0u;
			const uint fx0=(uint)((X-x0)*2), fy0=(uint)((Y-y0)*2), fz0_=is2D?0u:(uint)((Z-z0)*2);
			for(uint dz=0u; dz<cz; dz++) for(uint dy=0u; dy<2u; dy++) for(uint dx=0u; dx<2u; dx++) {
				const ulong n = idx3(fx0+dx, fy0+dy, fz0_+dz, fNx, fNy);
				if(!(ffl[n]&TYPE_S)) { ux += fu[n]; uy += fu[fN+n]; uz += fu[2ull*fN+n]; c++; }
			}
			if(c==0u) continue;
			ux /= (float)c; uy /= (float)c; uz /= (float)c;
			sum += (double)(sq(ux-pu[np])+sq(uy-pu[pN+np])+sq(uz-pu[2ull*pN+np]));
			cnt++;
		}
		return cnt>0ull ? (float)sqrt(sum/(double)cnt) : -1.0f;
	}

	// escribe u, rho y flags del nivel l como .vtk BINARIO con origen y espaciado SI CORRECTOS para superponer
	// todos los niveles en ParaView. path vacio -> bin/export/gloria/
	void write_vtk(const uint l, const string& path="") {
		Level& L = levels[l];
		LBM* lb = L.lbm;
		lb->u.read_from_device(); lb->rho.read_from_device(); lb->flags.read_from_device();
		const uint Nx=lb->get_Nx(), Ny=lb->get_Ny(), Nz=lb->get_Nz();
		const float dx = units_set ? L.u_si.si_x(1.0f) : 1.0f/(float)L.scale;
		const float cu = units_set ? L.u_si.si_u(1.0f) : 1.0f;
		const float cr = units_set ? L.u_si.si_rho(1.0f) : 1.0f;
		const Level& L0 = levels[0];
		const float dx0 = units_set ? L0.u_si.si_x(1.0f) : 1.0f;
		// origen: centro de la celda (0,0,0) del nivel l, en el sistema centrado del nivel 0, en metros
		const float ox = (L.g0x+0.5f/(float)L.scale-0.5f*(float)L0.lbm->get_Nx())*dx0;
		const float oy = (L.g0y+0.5f/(float)L.scale-0.5f*(float)L0.lbm->get_Ny())*dx0;
		const float oz = is2D ? 0.0f : (L.g0z+0.5f/(float)L.scale-0.5f*(float)L0.lbm->get_Nz())*dx0;
		const string dir = path=="" ? get_exe_path()+"export/gloria/" : path;
		const ulong N = (ulong)Nx*(ulong)Ny*(ulong)Nz;
		auto header = [&](const string& type, const uint dims) {
			return "# vtk DataFile Version 3.0\nFluidX3D-GLORIA nivel "+to_string(l)+"\nBINARY\nDATASET STRUCTURED_POINTS\n"
				"DIMENSIONS "+to_string(Nx)+" "+to_string(Ny)+" "+to_string(Nz)+"\n"
				"ORIGIN "+to_string(ox)+" "+to_string(oy)+" "+to_string(oz)+"\n"
				"SPACING "+to_string(dx)+" "+to_string(dx)+" "+to_string(dx)+"\n"
				"POINT_DATA "+to_string(N)+"\nSCALARS data "+type+" "+to_string(dims)+"\nLOOKUP_TABLE default\n";
		};
		const string suffix = "-L"+to_string(l)+"-t"+to_string(get_t())+".vtk";
		{ // u (3 componentes float, SI)
			const string filename = dir+"u"+suffix;
			create_folder(filename);
			std::ofstream file(filename, std::ios::out|std::ios::binary);
			const string h = header("float", 3u); file.write(h.c_str(), h.length());
			float* data = new float[3ull*N];
			parallel_for(N, [&](ulong i) { for(uint dd=0u; dd<3u; dd++) data[3ull*i+dd] = reverse_bytes(cu*lb->u[(ulong)dd*N+i]); }); // indexado SoA directo (valido con 1 dominio)
			file.write((char*)data, 3ull*N*sizeof(float)); file.close(); delete[] data;
			print_info("Archivo \""+filename+"\" guardado.");
		}
		{ // rho (float, SI)
			const string filename = dir+"rho"+suffix;
			create_folder(filename);
			std::ofstream file(filename, std::ios::out|std::ios::binary);
			const string h = header("float", 1u); file.write(h.c_str(), h.length());
			float* data = new float[N];
			parallel_for(N, [&](ulong i) { data[i] = reverse_bytes(cr*lb->rho[i]); });
			file.write((char*)data, N*sizeof(float)); file.close(); delete[] data;
			print_info("Archivo \""+filename+"\" guardado.");
		}
		{ // flags (uchar)
			const string filename = dir+"flags"+suffix;
			create_folder(filename);
			std::ofstream file(filename, std::ios::out|std::ios::binary);
			const string h = header("unsigned_char", 1u); file.write(h.c_str(), h.length());
			uchar* data = new uchar[N];
			parallel_for(N, [&](ulong i) { data[i] = lb->flags[i]; });
			file.write((char*)data, N*sizeof(uchar)); file.close(); delete[] data;
			print_info("Archivo \""+filename+"\" guardado.");
		}
	}
	void write_vtk_all(const string& path="") { for(uint l=0u; l<(uint)levels.size(); l++) write_vtk(l, path); }

	// imprime la tabla de niveles: resolucion, dx, dt, nu, tau, celdas y el "equivalente uniforme"
	void print_summary() {
		print_info("======================= GLORIA: niveles =======================");
		ulong total = 0ull;
		for(uint l=0u; l<(uint)levels.size(); l++) {
			Level& L = levels[l];
			const float tau = 3.0f*L.lbm->get_nu()+0.5f;
			string s = "nivel "+to_string(l)+": "+to_string(L.lbm->get_Nx())+"x"+to_string(L.lbm->get_Ny())+"x"+to_string(L.lbm->get_Nz());
			s += "  dx=dx0/"+to_string(L.scale);
			if(units_set) s += " ("+to_string(1000.0f*L.u_si.si_x(1.0f), 3u)+" mm)";
			s += "  nu="+to_string(L.lbm->get_nu(), 6u)+"  tau="+to_string(tau, 5u);
			print_info(s);
			total += L.lbm->get_N();
		}
		const Level& Lf = levels.back();
		const Level& L0 = levels.front();
		const ulong uniform = L0.lbm->get_N()*(ulong)(Lf.scale)*(ulong)(Lf.scale)*(is2D?1ull:(ulong)Lf.scale);
		print_info("celdas totales: "+to_string(total)+"  |  grilla uniforme equivalente a dx fino: "+to_string(uniform)+" ("+to_string((float)uniform/(float)total, 1u)+"x mas)");
		print_info("===============================================================");
	}
};
