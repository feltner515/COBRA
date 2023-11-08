# VerifyElementUniqueID

!syntax description /UserObjects/VerifyElementUniqueID

This object is used for debugging mesh issues.

!alert note
For distributed mesh, this will perform an `MPI_AllGather` operation, sending all ids to all processes, operation which may require a lot of memory on all processes.

!syntax parameters /UserObjects/VerifyElementUniqueID

!syntax inputs /UserObjects/VerifyElementUniqueID

!syntax children /UserObjects/VerifyElementUniqueID
