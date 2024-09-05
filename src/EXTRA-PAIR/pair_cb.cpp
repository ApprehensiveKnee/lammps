/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

// Contributing author: Edoardo Cabiati, Politecnico di Milano (edoardo.cabiati@mail.polimi.it)

#include "pair_cb.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "fix.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairCrowellBrown::PairCrowellBrown(LAMMPS *lmp) : Pair(lmp)
{
  writedata = 1;
}

PairCrowellBrown::~PairCrowellBrown()
{
    if (allocated) {
        memory->destroy(setflag);
        memory->destroy(cutsq);
        memory->destroy(cut);
        memory->destroy(epsilon);
        memory->destroy(sigma);
        memory->destroy(P_par);
        memory->destroy(P_perp);
    }
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairCrowellBrown::allocate()
{
    allocated = 1;
    int np1 = atom->ntypes + 1;

    memory->create(setflag, np1, np1, "pair:setflag");
    for (int i = 1; i < np1; i++)
        for (int j = i; j < np1; j++) setflag[i][j] = 0;

    memory->create(cutsq, np1, np1, "pair:cutsq");
    memory->create(cut, np1, np1, "pair:cut");
    memory->create(epsilon, np1, np1, "pair:epsilon");
    memory->create(sigma, np1, np1, "pair:sigma");
    memory->create(P_par, np1, np1, "pair:P_par");
    memory->create(P_perp, np1, np1, "pair:P_perp");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairCrowellBrown::settings(int narg, char **arg)
{
  if (narg != 1) error->all(FLERR, "Pair style crowell-brown must have exactly one argument");
  cut_global = utils::numeric(FLERR, arg[0], false, lmp);

  // reset per-type pair cutoffs that have been explicitly set previously

  if (allocated) {
    for (int i = 1; i <= atom->ntypes; i++)
      for (int j = i; j <= atom->ntypes; j++)
        if (setflag[i][j]) cut[i][j] = cut_global;
  }
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairCrowellBrown::coeff(int narg, char **arg)
{
    if (narg < 6 || narg > 7) error->all(FLERR, "Incorrect args for pair coefficients");
    if (!allocated) allocate();

    int ilo, ihi, jlo, jhi;
    utils::bounds(FLERR, arg[0], 1, atom->ntypes, ilo, ihi, error);
    utils::bounds(FLERR, arg[1], 1, atom->ntypes, jlo, jhi, error);

    double epsilon_one = utils::numeric(FLERR, arg[2], false, lmp);
    double sigma_one = utils::numeric(FLERR, arg[3], false, lmp);
    double P_par_one = utils::numeric(FLERR, arg[4], false, lmp);
    double P_perp_one = utils::numeric(FLERR, arg[5], false, lmp);
    double cut_one = cut_global;
    if (narg == 7) cut_one = utils::numeric(FLERR, arg[6], false, lmp);

    int count = 0;
    for (int i = ilo; i <= ihi; i++) {
        for (int j = MAX(jlo, i); j <= jhi; j++) {
        epsilon[i][j] = epsilon_one;
        sigma[i][j] = sigma_one;
        P_par[i][j] = P_par_one;
        P_perp[i][j] = P_perp_one;
        cut[i][j] = cut_one;
        setflag[i][j] = 1;
        count++;
        }
    }

    if (count == 0) error->all(FLERR, "Incorrect args for pair coefficients");
    }


/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairCrowellBrown::init_one(int i, int j)
{
    if (setflag[i][j] == 0) error->all(FLERR, "All pair coeffs are not set");

    epsilon[j][i] = epsilon[i][j];
    sigma[j][i] = sigma[i][j];
    P_par[j][i] = P_par[i][j];
    P_perp[j][i] = P_perp[i][j];

    return cut[i][j];
}

/* ---------------------------------------------------------------------- */

void PairCrowellBrown::compute(int eflag, int vflag)
{
    int i, j, ii, jj, inum, jnum, itype, jtype;
    double xtmp, ytmp, ztmp, delx, dely, delz, evdwl, fpair_xy, fpair_z, fpair;
    double rsq, r, dr, factor_lj, afct, bfct;
    double cb1, cb2, cb3;
    double cos_theta;
    int *ilist, *jlist, *numneigh, **firstneigh;

    evdwl = 0.0;
    ev_init(eflag, vflag);

    double **x = atom->x;
    double **f = atom->f;
    int *type = atom->type;
    int nlocal = atom->nlocal;
    double *special_lj = force->special_lj;
    int newton_pair = force->newton_pair;

    inum = list->inum;
    ilist = list->ilist;
    numneigh = list->numneigh;
    firstneigh = list->firstneigh;

    // loop over neighbors of my atoms

    for (ii = 0; ii < inum; ii++) {
        i = ilist[ii];
        xtmp = x[i][0];
        ytmp = x[i][1];
        ztmp = x[i][2];
        itype = type[i];
        jlist = firstneigh[i];
        jnum = numneigh[i];
        for (jj = 0; jj < jnum; jj++) {
            j = jlist[jj];
            factor_lj = special_lj[sbmask(j)];
            j &= NEIGHMASK;

            delx = xtmp - x[j][0];
            dely = ytmp - x[j][1];
            delz = ztmp - x[j][2];
            rsq = delx * delx + dely * dely + delz * delz;
            jtype = type[j];

            if (rsq < cutsq[itype][jtype]) {
                r = sqrt(rsq);
                //Compute the angle between the vector connecting the two atoms and the z-axis
                //to later use for the potential computation

                //                z
                //   _____________|________________________________________  carbon platelet
                //                |
                //                |
                // hydrogen  o    |
                //            \   |
                //             \  |  normal to the graphite plane
                //              \ |  
                //   ____________\|________________________________________   carbon platelet
                //                |\
                //                | \
                //                |  \
                //                | V \
                //               theta angle

                cos_theta = delz/r;
                // The Crowell-Brown potential is defined as:
                // V_CB(r) = epsilon_CB*{(sigma_CB/r)^12 - (sigma_CB/r)^6*[3*(P_par-P_perp)*cos^2(theta) + (P_par + 5*P_perp)]/[4*P_par + 2*P_perp]}
                afct = pow(sigma[itype][jtype] / r, 12);
                bfct = -pow(sigma[itype][jtype] / r, 6) * (3 * (P_par[itype][jtype] - P_perp[itype][jtype]) * cos_theta * cos_theta + (P_par[itype][jtype] + 5 * P_perp[itype][jtype])) / (4 * P_par[itype][jtype] + 2 * P_perp[itype][jtype]);
                cb1 = afct*12/(r*r);
                cb2 = bfct*6/(r*r);
                cb3 = pow(sigma[itype][jtype] / r, 6) * 6 *(P_par[itype][jtype] - P_perp[itype][jtype])*(delz/pow(r,4))/((4 * P_par[itype][jtype] + 2 * P_perp[itype][jtype]));

                fpair_xy = epsilon[itype][jtype] * (cb1 + cb2 - cb3*delz);
                fpair_z = epsilon[itype][jtype] * (cb1*delz + cb2*delz + cb3*(r*r-delz*delz));
                fpair_xy *= factor_lj;
                fpair_z *= factor_lj;
                fpair = sqrt(pow(fpair_xy*delx, 2)+ pow(fpair_xy*dely, 2) + pow(fpair_z, 2));

                f[i][0] += delx * fpair_xy;
                f[i][1] += dely * fpair_xy;
                f[i][2] += fpair_z;
                if (newton_pair || j < nlocal) {
                    f[j][0] -= delx * fpair_xy;
                    f[j][1] -= dely * fpair_xy;
                    f[j][2] -= fpair_z;
                };

                
                if (eflag) 
                {
                  // Compute the offset for each pair of atoms if needed
                  double cut_ij = sqrt(cutsq[itype][jtype]);
                  double offset = (pow(sigma[itype][jtype]/cut_ij, 12) - pow(sigma[itype][jtype]/cut_ij, 6) * (3 * (P_par[itype][jtype] - P_perp[itype][jtype]) * cos_theta * cos_theta + (P_par[itype][jtype] + 5 * P_perp[itype][jtype])) / (4 * P_par[itype][jtype] + 2 * P_perp[itype][jtype]));
                  evdwl = factor_lj * epsilon[itype][jtype] *(afct + bfct - offset);
                }
                
                if (evflag) ev_tally(i, j, nlocal, newton_pair, evdwl, 0.0, fpair, delx, dely, delz);
            }
        }
    }
    if (vflag_fdotr) virial_fdotr_compute();
}

/* ---------------------------------------------------------------------- */

double PairCrowellBrown::single(int i, int j, int itype, int jtype, double rsq,
                             double /*factor_coul*/, double factor_lj, double &fforce)
{
    double delx, dely, delz;
    double r, cos_theta, fpair_xy, fpair_z;
    double afct, bfct;
    double cb1, cb2, cb3;

    double **x = atom->x;

    r = sqrt(rsq);
    delx = x[i][0] - x[j][0];
    dely = x[i][1] - x[j][1];
    delz = x[i][2] - x[j][2];

    //Compute the angle between the vector connecting the two atoms and the z-axis
    cos_theta = delz/r;

    afct = pow(sigma[itype][jtype] / r, 12);
    bfct = -pow(sigma[itype][jtype] / r, 6) * (3 * (P_par[itype][jtype] - P_perp[itype][jtype]) * cos_theta * cos_theta + (P_par[itype][jtype] + 5 * P_perp[itype][jtype])) / (4 * P_par[itype][jtype] + 2 * P_perp[itype][jtype]);
    cb1 = afct*12/(r*r);
    cb2 = bfct*6/(r*r);
    cb3 = pow(sigma[itype][jtype] / r, 6) * 6 *(P_par[itype][jtype] - P_perp[itype][jtype])*(delz/pow(r,4))/((4 * P_par[itype][jtype] + 2 * P_perp[itype][jtype]));

    fpair_xy = epsilon[itype][jtype] * (cb1 + cb2 - cb3*delz);
    fpair_z = epsilon[itype][jtype] * (cb1*delz + cb2*delz + cb3*(r*r-delz*delz));
    fforce = sqrt(pow(fpair_xy*delx, 2)+ pow(fpair_xy*dely, 2) + pow(fpair_z, 2));
    fforce *= factor_lj;

    // Compute the offset for each pair of atoms if needed
    double cut_ij = cut[itype][jtype];
    double offset =(pow(sigma[itype][jtype] / cut_ij, 12) - pow(sigma[itype][jtype] / cut_ij, 6) * (3 * (P_par[itype][jtype] - P_perp[itype][jtype]) * cos_theta * cos_theta + (P_par[itype][jtype] + 5 * P_perp[itype][jtype])) / (4 * P_par[itype][jtype] + 2 * P_perp[itype][jtype]));
    return factor_lj * ( epsilon[itype][jtype]*(afct + bfct - offset));
    }

    /* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairCrowellBrown::write_restart(FILE *fp)
{
  write_restart_settings(fp);

  int i, j;
  for (i = 1; i <= atom->ntypes; i++) {
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j], sizeof(int), 1, fp);
      if (setflag[i][j]) {
        fwrite(&epsilon[i][j], sizeof(double), 1, fp);
        fwrite(&sigma[i][j], sizeof(double), 1, fp);
        fwrite(&P_par[i][j], sizeof(double), 1, fp);
        fwrite(&P_perp[i][j], sizeof(double), 1, fp);
        fwrite(&cut[i][j], sizeof(double), 1, fp);
      }
    }
  }
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairCrowellBrown::write_restart_settings(FILE *fp)
{
  fwrite(&cut_global, sizeof(double), 1, fp);
  fwrite(&offset_flag, sizeof(int), 1, fp);
  fwrite(&mix_flag, sizeof(int), 1, fp);
}


/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairCrowellBrown::read_restart(FILE *fp)
{
  read_restart_settings(fp);

  allocate();

  int i, j;
  int me = comm->me;
  for (i = 1; i <= atom->ntypes; i++) {
    for (j = i; j <= atom->ntypes; j++) {
      if (me == 0) utils::sfread(FLERR, &setflag[i][j], sizeof(int), 1, fp, nullptr, error);
      MPI_Bcast(&setflag[i][j], 1, MPI_INT, 0, world);
      if (setflag[i][j]) {
        if (me == 0) {
            utils::sfread(FLERR, &epsilon[i][j], sizeof(double), 1, fp, nullptr, error);
            utils::sfread(FLERR, &sigma[i][j], sizeof(double), 1, fp, nullptr, error);
            utils::sfread(FLERR, &P_par[i][j], sizeof(double), 1, fp, nullptr, error);
            utils::sfread(FLERR, &P_perp[i][j], sizeof(double), 1, fp, nullptr, error);
            utils::sfread(FLERR, &cut[i][j], sizeof(double), 1, fp, nullptr, error);
        }
        MPI_Bcast(&epsilon[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&sigma[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&P_par[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&P_perp[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&cut[i][j], 1, MPI_DOUBLE, 0, world);
      }
    }
  }
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairCrowellBrown::read_restart_settings(FILE *fp)
{
  if (comm->me == 0) {
    utils::sfread(FLERR, &cut_global, sizeof(double), 1, fp, nullptr, error);
    utils::sfread(FLERR, &offset_flag, sizeof(int), 1, fp, nullptr, error);
    utils::sfread(FLERR, &mix_flag, sizeof(int), 1, fp, nullptr, error);
  }
  MPI_Bcast(&cut_global, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&offset_flag, 1, MPI_INT, 0, world);
  MPI_Bcast(&mix_flag, 1, MPI_INT, 0, world);
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void PairCrowellBrown::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    fprintf(fp, "%d %g %g %g %g\n", i, epsilon[i][i], sigma[i][i], P_par[i][i], P_perp[i][i]);
}

/* ----------------------------------------------------------------------
   proc 0 writes all pairs to data file
------------------------------------------------------------------------- */

void PairCrowellBrown::write_data_all(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = i; j <= atom->ntypes; j++)
      fprintf(fp, "%d %d %g %g %g %g %g\n", i, j, epsilon[i][j], sigma[i][j],  P_par[i][j], P_perp[i][j] , cut[i][j]);
}

/* ---------------------------------------------------------------------- */

void *PairCrowellBrown::extract(const char *str, int &dim)
{
    dim = 2;
    if (strcmp(str, "epsilon") == 0) return (void *) epsilon;
    if (strcmp(str, "sigma") == 0) return (void *) sigma;
    if (strcmp(str, "P_par") == 0) return (void *) P_par;
    if (strcmp(str, "P_perp") == 0) return (void *) P_perp;
  return nullptr;
}