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

#include "pair_sg.h"

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

PairSilveraGoldman::PairSilveraGoldman(LAMMPS *lmp) : Pair(lmp)
{
  writedata = 1;
}

PairSilveraGoldman::~PairSilveraGoldman()
{
    if (allocated) {
        memory->destroy(setflag);
        memory->destroy(cutsq);
        memory->destroy(cut);
        memory->destroy(rc);
        memory->destroy(c9);
        memory->destroy(c10);
        memory->destroy(c8);
        memory->destroy(c6);
        memory->destroy(gamma);
        memory->destroy(beta);
        memory->destroy(alpha);
        memory->destroy(offset);
    }
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairSilveraGoldman::allocate()
{
    allocated = 1;
    int np1 = atom->ntypes + 1;

    memory->create(setflag, np1, np1, "pair:setflag");
    for (int i = 1; i < np1; i++)
        for (int j = i; j < np1; j++) setflag[i][j] = 0;

    memory->create(cutsq, np1, np1, "pair:cutsq");
    memory->create(cut, np1, np1, "pair:cut");
    memory->create(rc, np1, np1, "pair:rc");
    memory->create(c9, np1, np1, "pair:c9");
    memory->create(c10, np1, np1, "pair:c10");
    memory->create(c8, np1, np1, "pair:c8");
    memory->create(c6, np1, np1, "pair:c6");
    memory->create(gamma, np1, np1, "pair:gamma");
    memory->create(beta, np1, np1, "pair:beta");
    memory->create(alpha, np1, np1, "pair:alpha");
    memory->create(offset, np1, np1, "pair:offset");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairSilveraGoldman::settings(int narg, char **arg)
{
  if (narg != 1) error->all(FLERR, "Pair style silvera-goldman must have exactly one argument");
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

void PairSilveraGoldman::coeff(int narg, char **arg)
{
    if (narg < 10 || narg > 11) error->all(FLERR, "Incorrect args for pair coefficients");
    if (!allocated) allocate();

    int ilo, ihi, jlo, jhi;
    utils::bounds(FLERR, arg[0], 1, atom->ntypes, ilo, ihi, error);
    utils::bounds(FLERR, arg[1], 1, atom->ntypes, jlo, jhi, error);

    double alpha_one = utils::numeric(FLERR, arg[2], false, lmp);
    double sigma_one = utils::numeric(FLERR, arg[3], false, lmp);
    double gamma_one = utils::numeric(FLERR, arg[4], false, lmp);
    double c6_one = utils::numeric(FLERR, arg[5], false, lmp);
    double c8_one = utils::numeric(FLERR, arg[6], false, lmp);
    double c9_one = utils::numeric(FLERR, arg[7], false, lmp);
    double c10_one = utils::numeric(FLERR, arg[8], false, lmp);
    double rc_one = utils::numeric(FLERR, arg[9], false, lmp);
    double cut_one = cut_global;
    if (narg == 11) cut_one = utils::numeric(FLERR, arg[10], false, lmp);

    int count = 0;
    for (int i = ilo; i <= ihi; i++) {
        for (int j = MAX(jlo, i); j <= jhi; j++) {
        alpha[i][j] = alpha_one;
        beta[i][j] = sigma_one;
        gamma[i][j] = gamma_one;
        c6[i][j] = c6_one;
        c8[i][j] = c8_one;
        c9[i][j] = c9_one;
        c10[i][j] = c10_one;
        rc[i][j] = rc_one;
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

double PairSilveraGoldman::init_one(int i, int j)
{
    if (setflag[i][j] == 0) error->all(FLERR, "All pair coeffs are not set");

    if (offset_flag) {
        double fc = exp(-(rc[i][j]/cut[i][j]-1)*(rc[i][j]/cut[i][j]-1));
        double r6 = pow(rc[i][j],6);
        double r8 = r6*rc[i][j]*rc[i][j];
        double r9 = r8*rc[i][j];
        double r10 = r9*rc[i][j];
        if(cut[i][j] > rc[i][j])  fc = 1.0;
        offset[i][j] =
            exp(alpha[i][j]-beta[i][j]*cut[i][j]-gamma[i][j]*cut[i][j]*cut[i][j]) - (c6[i][j]/r6+c8[i][j]/r8+c10[i][j]/r10)*fc + (c9[i][j]/r9)*fc;
    } else
        offset[i][j] = 0.0;

    alpha[j][i] = alpha[i][j];
    beta[j][i] = beta[i][j];
    gamma[j][i] = gamma[i][j];
    c6[j][i] = c6[i][j];
    c8[j][i] = c8[i][j];
    c9[j][i] = c9[i][j];
    c10[j][i] = c10[i][j];
    rc[j][i] = rc[i][j];
    offset[j][i] = offset[i][j];

    return cut[i][j];
}

/* ---------------------------------------------------------------------- */

void PairSilveraGoldman::compute(int eflag, int vflag)
{
    int i, j, ii, jj, inum, jnum, itype, jtype;
    double xtmp, ytmp, ztmp, delx, dely, delz, evdwl, fpair;
    double rsq, r, dr, factor_lj, afct, bfct;
    double r6, r8, r9, r10, r11, r12;
    double sg1, sg2;
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
          r6 = pow(r,6);
          r8 = r6*r*r;
          r9 = r8*r;
          r10 = r9*r;
          r11 = r10*r;
          r12 = r11*r;
          afct = exp(- gamma[itype][jtype]*r*r - beta[itype][jtype]*r + alpha[itype][jtype]);
          sg1 = afct*(beta[itype][jtype]/r + 2*gamma[itype][jtype]);
          sg2 = -6*c6[itype][jtype]/r8 - 8*c8[itype][jtype]/r10 - 10*c10[itype][jtype]/r12 + 9*c9[itype][jtype]/r11;
          double fc = exp(-(rc[itype][jtype]/r - 1)*(rc[itype][jtype]/r - 1));
          if (r <= rc[itype][jtype])
          {
              bfct = - (c6[itype][jtype]/r6 + c8[itype][jtype]/r8 + c10[itype][jtype]/r10*fc - c9[itype][jtype]/r9)*fc;
              sg2 = sg2*fc + bfct*fc*(rc[itype][jtype]/r - 1)*(2*rc[itype][jtype]/pow(r,3));

          }
          else
          {
              fc = 1.0;
              bfct = - (c6[itype][jtype]/r6 + c8[itype][jtype]/r8 + c10[itype][jtype]/r10*fc - c9[itype][jtype]/r9)*fc;
          };
          
          fpair = sg1 + sg2;
          fpair *= factor_lj;
          // Since the parameters for the SG potential are given for atomic units, convert to eV/Å and then to (kcal/mol)/Å
          fpair *= 23.06  * 51.421;

          f[i][0] += delx * fpair;
          f[i][1] += dely * fpair;
          f[i][2] += delz * fpair;
          if (newton_pair || j < nlocal) {
            f[j][0] -= delx * fpair;
            f[j][1] -= dely * fpair;
            f[j][2] -= delz * fpair;
          };

          if (eflag) evdwl = factor_lj * 23.06 * 27.211 *(afct + bfct - offset[itype][jtype]);
          if (evflag) ev_tally(i, j, nlocal, newton_pair, evdwl, 0.0, fpair, delx, dely, delz);
        }
      }
    }
    if (vflag_fdotr) virial_fdotr_compute();
}

/* ---------------------------------------------------------------------- */

double PairSilveraGoldman::single(int /*i*/, int /*j*/, int itype, int jtype, double rsq,
                             double /*factor_coul*/, double factor_lj, double &fforce)
{
    double r, sg1, sg2, fpair;
    double r6, r8, r9, r10, r11, r12;
    double afct, bfct;

    r = sqrt(rsq);
    r6 = pow(r,6);
    r8 = r6*r*r;
    r9 = r8*r;
    r10 = r9*r;
    r11 = r10*r;
    r12 = r11*r;
    sg1 = exp(alpha[itype][jtype] - beta[itype][jtype]*r - gamma[itype][jtype]*r*r)*(beta[itype][jtype]/r + 2*gamma[itype][jtype]);
    afct = exp(-gamma[itype][jtype]*r*r - beta[itype][jtype]*r + alpha[itype][jtype]);
    sg2 = -6*c6[itype][jtype]/r8 - 8*c8[itype][jtype]/r10 - 10*c10[itype][jtype]/r12 + 9*c9[itype][jtype]/r11;
    double fc = exp(-(rc[itype][jtype]/r - 1)*(rc[itype][jtype]/r - 1));
    if (r <= rc[itype][jtype])
    {
        bfct = - (c6[itype][jtype]/r6 + c8[itype][jtype]/r8 + c10[itype][jtype]/r10*fc - c9[itype][jtype]/r9)*fc;
        sg2 = sg2*fc + bfct*exp(-(rc[itype][jtype]/r - 1)*(rc[itype][jtype]/r - 1))*(rc[itype][jtype]/r - 1)*(2*rc[itype][jtype]/pow(r,3));

    }
    else
    {
        fc = 1.0;
        bfct = - (c6[itype][jtype]/r6 + c8[itype][jtype]/r8 + c10[itype][jtype]/r10*fc - c9[itype][jtype]/r9)*fc;

    };
    fpair = sg1 + sg2;

    fforce = factor_lj *fpair;
    // Since the parameters for the SG potential are given for atomic units, convert to eV/Å and then to (kcal/mol)/Å
    fforce *= 23.06  * 51.421;
    return factor_lj * 23.06 * 27.211 *(afct + bfct - offset[itype][jtype]);
    }

    /* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairSilveraGoldman::write_restart(FILE *fp)
{
  write_restart_settings(fp);

  int i, j;
  for (i = 1; i <= atom->ntypes; i++) {
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j], sizeof(int), 1, fp);
      if (setflag[i][j]) {
        fwrite(&alpha[i][j], sizeof(double), 1, fp);
        fwrite(&beta[i][j], sizeof(double), 1, fp);
        fwrite(&gamma[i][j], sizeof(double), 1, fp);
        fwrite(&c6[i][j], sizeof(double), 1, fp);
        fwrite(&c8[i][j], sizeof(double), 1, fp);
        fwrite(&c9[i][j], sizeof(double), 1, fp);
        fwrite(&c10[i][j], sizeof(double), 1, fp);
        fwrite(&rc[i][j], sizeof(double), 1, fp);
        fwrite(&cut[i][j], sizeof(double), 1, fp);
      }
    }
  }
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairSilveraGoldman::write_restart_settings(FILE *fp)
{
  fwrite(&cut_global, sizeof(double), 1, fp);
  fwrite(&offset_flag, sizeof(int), 1, fp);
  fwrite(&mix_flag, sizeof(int), 1, fp);
}


/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairSilveraGoldman::read_restart(FILE *fp)
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
            utils::sfread(FLERR, &alpha[i][j], sizeof(double), 1, fp, nullptr, error);
            utils::sfread(FLERR, &beta[i][j], sizeof(double), 1, fp, nullptr, error);
            utils::sfread(FLERR, &gamma[i][j], sizeof(double), 1, fp, nullptr, error);
            utils::sfread(FLERR, &c6[i][j], sizeof(double), 1, fp, nullptr, error);
            utils::sfread(FLERR, &c8[i][j], sizeof(double), 1, fp, nullptr, error);
            utils::sfread(FLERR, &c9[i][j], sizeof(double), 1, fp, nullptr, error);
            utils::sfread(FLERR, &c10[i][j], sizeof(double), 1, fp, nullptr, error);
            utils::sfread(FLERR, &rc[i][j], sizeof(double), 1, fp, nullptr, error);
            utils::sfread(FLERR, &cut[i][j], sizeof(double), 1, fp, nullptr, error);
        }
        MPI_Bcast(&alpha[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&beta[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&gamma[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&c6[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&c8[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&c9[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&c10[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&rc[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&cut[i][j], 1, MPI_DOUBLE, 0, world);
      }
    }
  }
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairSilveraGoldman::read_restart_settings(FILE *fp)
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

void PairSilveraGoldman::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    fprintf(fp, "%d %g %g %g %g %g %g %g %g\n", i, alpha[i][i], beta[i][i], gamma[i][i], c6[i][i], c8[i][i], c9[i][i], c10[i][i], rc[i][i]);
}

/* ----------------------------------------------------------------------
   proc 0 writes all pairs to data file
------------------------------------------------------------------------- */

void PairSilveraGoldman::write_data_all(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = i; j <= atom->ntypes; j++)
      fprintf(fp, "%d %d %g %g %g %g %g %g %g %g %g\n", i, j, alpha[i][i], beta[i][i], gamma[i][i], c6[i][i], c8[i][i], c9[i][i], c10[i][i], rc[i][i], cut[i][j]);
}

/* ---------------------------------------------------------------------- */

void *PairSilveraGoldman::extract(const char *str, int &dim)
{
    dim = 2;
    if (strcmp(str, "alpha") == 0) return (void *) alpha;
    if (strcmp(str, "beta") == 0) return (void *) beta;
    if (strcmp(str, "gamma") == 0) return (void *) gamma;
    if (strcmp(str, "c6") == 0) return (void *) c6;
    if (strcmp(str, "c8") == 0) return (void *) c8;
    if (strcmp(str, "c9") == 0) return (void *) c9;
    if (strcmp(str, "c10") == 0) return (void *) c10;
    if (strcmp(str, "rc") == 0) return (void *) rc;
  return nullptr;
}