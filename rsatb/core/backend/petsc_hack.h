#ifndef DMPLEX_HACK
#define DMPLEX_HACK

#include <petsc/private/dmpleximpl.h>
#include <petsc/private/hashseti.h>
#include <petscsf.h>
#include <petscdmplextransform.h>
#include <petscdmlabelephemeral.h>
#include <petsc/private/kernels/blockmatmult.h>
#include <petsc/private/kernels/blockinvert.h>
#ifdef PETSC_HAVE_UNISTD_H
  #include <unistd.h>
#endif
#include <errno.h>


inline PetscErrorCode HACKPetscSFCreateByMatchingIndices(PetscLayout layout, PetscInt numRootIndices, const PetscInt *rootIndices, const PetscInt *rootLocalIndices, PetscInt rootLocalOffset, PetscInt numLeafIndices, const PetscInt *leafIndices, const PetscInt *leafLocalIndices, PetscInt leafLocalOffset, PetscSF *sfA, PetscSF *sf)
{
  MPI_Comm     comm = layout->comm;
  PetscMPIInt  size, rank;
  PetscSF      sf1;
  PetscSFNode *owners, *buffer, *iremote;
  PetscInt    *ilocal, nleaves, N, n, i, vStart, vEnd;
#if defined(PETSC_USE_DEBUG)
  PetscInt N1;
#endif
  PetscBool flag;
  
  PetscFunctionBegin;
  if (rootIndices) PetscAssertPointer(rootIndices, 3);
  if (rootLocalIndices) PetscAssertPointer(rootLocalIndices, 4);
  if (leafIndices) PetscAssertPointer(leafIndices, 7);
  if (leafLocalIndices) PetscAssertPointer(leafLocalIndices, 8);
  if (sfA) PetscAssertPointer(sfA, 10);
  PetscAssertPointer(sf, 11);
  PetscCheck(numRootIndices >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "numRootIndices (%" PetscInt_FMT ") must be non-negative", numRootIndices);
  PetscCheck(numLeafIndices >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "numLeafIndices (%" PetscInt_FMT ") must be non-negative", numLeafIndices);
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(PetscLayoutSetUp(layout));
  PetscCall(PetscLayoutGetSize(layout, &N));
  PetscCall(PetscLayoutGetLocalSize(layout, &n));
  PetscCall(PetscLayoutGetRange(layout, &vStart, &vEnd));
  flag = (PetscBool)(leafIndices == rootIndices);
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &flag, 1, MPI_C_BOOL, MPI_LAND, comm));
  PetscCheck(!flag || numLeafIndices == numRootIndices, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "leafIndices == rootIndices, but numLeafIndices (%" PetscInt_FMT ") != numRootIndices(%" PetscInt_FMT ")", numLeafIndices, numRootIndices);
#if defined(PETSC_USE_DEBUG)
  N1 = PETSC_INT_MIN;
  for (i = 0; i < numRootIndices; i++)
    if (rootIndices[i] > N1) N1 = rootIndices[i];
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &N1, 1, MPIU_INT, MPI_MAX, comm));
  PetscCheck(N1 < N, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Max. root index (%" PetscInt_FMT ") out of layout range [0,%" PetscInt_FMT ")", N1, N);
  if (!flag) {
    N1 = PETSC_INT_MIN;
    for (i = 0; i < numLeafIndices; i++)
      if (leafIndices[i] > N1) N1 = leafIndices[i];
    PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &N1, 1, MPIU_INT, MPI_MAX, comm));
    PetscCheck(N1 < N, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Max. leaf index (%" PetscInt_FMT ") out of layout range [0,%" PetscInt_FMT ")", N1, N);
  }
#endif
  /* Reduce: owners -> buffer */
  PetscCall(PetscMalloc1(n, &buffer));
  PetscCall(PetscSFCreate(comm, &sf1));
  PetscCall(PetscSFSetFromOptions(sf1));
  PetscCall(PetscSFSetGraphLayout(sf1, layout, numRootIndices, NULL, PETSC_OWN_POINTER, rootIndices));
  PetscCall(PetscMalloc1(numRootIndices, &owners));
  for (i = 0; i < numRootIndices; ++i) {
    if(rootIndices[i]>=vStart && rootIndices[i]<vEnd){
        owners[i].rank  = rank;
        owners[i].index = rootLocalOffset + (rootLocalIndices ? rootLocalIndices[i] : i);
    } else {
        owners[i].rank  = -1;
        owners[i].index = -1;
    }
  }
  for (i = 0; i < n; ++i) {
    buffer[i].index = -1;
    buffer[i].rank  = -1;
  }
  PetscCall(PetscSFReduceBegin(sf1, MPIU_SF_NODE, owners, buffer, MPI_MAXLOC));
  PetscCall(PetscSFReduceEnd(sf1, MPIU_SF_NODE, owners, buffer, MPI_MAXLOC));
  /* Bcast: buffer -> owners */
  if (!flag) {
    /* leafIndices is different from rootIndices */
    PetscCall(PetscFree(owners));
    PetscCall(PetscSFSetGraphLayout(sf1, layout, numLeafIndices, NULL, PETSC_OWN_POINTER, leafIndices));
    PetscCall(PetscMalloc1(numLeafIndices, &owners));
  }
  PetscCall(PetscSFBcastBegin(sf1, MPIU_SF_NODE, buffer, owners, MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(sf1, MPIU_SF_NODE, buffer, owners, MPI_REPLACE));
  for (i = 0; i < numLeafIndices; ++i) PetscCheck(owners[i].rank >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Global point %" PetscInt_FMT " was unclaimed", leafIndices[i]);
  PetscCall(PetscFree(buffer));
  if (sfA) {
    *sfA = sf1;
  } else PetscCall(PetscSFDestroy(&sf1));
  /* Create sf */
  if (flag && rootLocalIndices == leafLocalIndices && leafLocalOffset == rootLocalOffset) {
    /* leaf space == root space */
    for (i = 0, nleaves = 0; i < numLeafIndices; ++i)
      if (owners[i].rank != rank) ++nleaves;
    PetscCall(PetscMalloc1(nleaves, &ilocal));
    PetscCall(PetscMalloc1(nleaves, &iremote));
    for (i = 0, nleaves = 0; i < numLeafIndices; ++i) {
      if (owners[i].rank != rank) {
        ilocal[nleaves]        = leafLocalOffset + i;
        iremote[nleaves].rank  = owners[i].rank;
        iremote[nleaves].index = owners[i].index;
        ++nleaves;
      }
    }
    PetscCall(PetscFree(owners));
  } else {
    nleaves = numLeafIndices;
    PetscCall(PetscMalloc1(nleaves, &ilocal));
    for (i = 0; i < nleaves; ++i) ilocal[i] = leafLocalOffset + (leafLocalIndices ? leafLocalIndices[i] : i);
    iremote = owners;
  }
  PetscCall(PetscSFCreate(comm, sf));
  PetscCall(PetscSFSetFromOptions(*sf));
  PetscCall(PetscSFSetGraph(*sf, rootLocalOffset + numRootIndices, nleaves, ilocal, PETSC_OWN_POINTER, iremote, PETSC_OWN_POINTER));
  PetscFunctionReturn(PETSC_SUCCESS);
}


inline PetscErrorCode HACKDMPlexBuildFromCellListParallel(DM dm, PetscInt numCells, PetscInt numVertices, PetscInt NVertices, PetscInt numCorners, const PetscInt cells[], PeOp PetscSF *vertexSF, PeOp PetscInt *verticesAdjSaved[])
{
    PetscSF     sfPoint;
    PetscLayout layout;
    PetscInt    numVerticesAdj, *verticesAdj, *cones, c, p;

    PetscFunctionBegin;
    PetscCall(PetscLogEventBegin(DMPLEX_BuildFromCellList, dm, 0, 0, 0));
    /* Get/check global number of vertices */
    {
        PetscInt       NVerticesInCells, i;
        const PetscInt len = numCells * numCorners;

        /* NVerticesInCells = max(cells) + 1 */
        NVerticesInCells = PETSC_INT_MIN;
        for (i = 0; i < len; i++)
        if (cells[i] > NVerticesInCells) NVerticesInCells = cells[i];
        ++NVerticesInCells;
        PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &NVerticesInCells, 1, MPIU_INT, MPI_MAX, PetscObjectComm((PetscObject)dm)));

        if (numVertices == PETSC_DECIDE && NVertices == PETSC_DECIDE) NVertices = NVerticesInCells;
        else
        PetscCheck(NVertices == PETSC_DECIDE || NVertices >= NVerticesInCells, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Specified global number of vertices %" PetscInt_FMT " must be greater than or equal to the number of vertices in cells %" PetscInt_FMT, NVertices, NVerticesInCells);
    }
    /* Count locally unique vertices */
    {
        PetscHSetI vhash;
        PetscInt   off = 0;

        PetscCall(PetscHSetICreate(&vhash));
        for (c = 0; c < numCells; ++c) {
        for (p = 0; p < numCorners; ++p) PetscCall(PetscHSetIAdd(vhash, cells[c * numCorners + p]));
        }
        PetscCall(PetscHSetIGetSize(vhash, &numVerticesAdj));
        if (!verticesAdjSaved) PetscCall(PetscMalloc1(numVerticesAdj, &verticesAdj));
        else verticesAdj = *verticesAdjSaved;
        PetscCall(PetscHSetIGetElems(vhash, &off, verticesAdj));
        PetscCall(PetscHSetIDestroy(&vhash));
        PetscCheck(off == numVerticesAdj, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid number of local vertices %" PetscInt_FMT " should be %" PetscInt_FMT, off, numVerticesAdj);
    }
    PetscCall(PetscSortInt(numVerticesAdj, verticesAdj));
    /* Create cones */
    PetscCall(DMPlexSetChart(dm, 0, numCells + numVerticesAdj));
    for (c = 0; c < numCells; ++c) PetscCall(DMPlexSetConeSize(dm, c, numCorners));
    PetscCall(DMSetUp(dm));
    PetscCall(DMPlexGetCones(dm, &cones));
    for (c = 0; c < numCells; ++c) {
        for (p = 0; p < numCorners; ++p) {
        const PetscInt gv = cells[c * numCorners + p];
        PetscInt       lv;

        /* Positions within verticesAdj form 0-based local vertex numbering;
            we need to shift it by numCells to get correct DAG points (cells go first) */
        PetscCall(PetscFindInt(gv, numVerticesAdj, verticesAdj, &lv));
        PetscCheck(lv >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Could not find global vertex %" PetscInt_FMT " in local connectivity", gv);
        cones[c * numCorners + p] = lv + numCells;
        }
    }
    /* Build point sf */
    PetscCall(PetscLayoutCreate(PetscObjectComm((PetscObject)dm), &layout));
    PetscCall(PetscLayoutSetSize(layout, NVertices));
    PetscCall(PetscLayoutSetLocalSize(layout, numVertices));
    PetscCall(PetscLayoutSetBlockSize(layout, 1));
    PetscCall(PetscLayoutSetUp(layout));

    PetscCall(HACKPetscSFCreateByMatchingIndices(
         layout,
         numVerticesAdj, verticesAdj, NULL, numCells, 
         numVerticesAdj, verticesAdj, NULL, numCells, vertexSF, &sfPoint));

    PetscCall(PetscLayoutDestroy(&layout));
    if (!verticesAdjSaved) PetscCall(PetscFree(verticesAdj));
    PetscCall(PetscObjectSetName((PetscObject)sfPoint, "point SF"));
    if (dm->sf) {
        const char *prefix;

        PetscCall(PetscObjectGetOptionsPrefix((PetscObject)dm->sf, &prefix));
        PetscCall(PetscObjectSetOptionsPrefix((PetscObject)sfPoint, prefix));
    }
    PetscCall(DMSetPointSF(dm, sfPoint));
    PetscCall(PetscSFDestroy(&sfPoint));
    if (vertexSF) PetscCall(PetscObjectSetName((PetscObject)*vertexSF, "Vertex Ownership SF"));
    /* Fill in the rest of the topology structure */
    PetscCall(DMPlexSymmetrize(dm));
    PetscCall(DMPlexStratify(dm));
    PetscCall(PetscLogEventEnd(DMPLEX_BuildFromCellList, dm, 0, 0, 0));
    PetscFunctionReturn(PETSC_SUCCESS);
}

#endif