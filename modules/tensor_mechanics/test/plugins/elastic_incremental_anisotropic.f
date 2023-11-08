****************************************************************************************
**  UMAT, FOR ABAQUS/STANDARD INCORPORATING ISOTROPIC ELASTICITY                      **
****************************************************************************************
****************************************************************************************
**
**
**
      SUBROUTINE UMAT(STRESS,STATEV,DDSDDE,SSE,SPD,SCD,
     1     RPL,DDSDDT,DRPLDE,DRPLDT,
     2     STRAN,DSTRAN,TIME,DTIME,TEMP,DTEMP,PREDEF,DPRED,CMNAME,
     3     NDI,NSHR,NTENS,NSTATV,PROPS,NPROPS,COORDS,DROT,PNEWDT,
     4     CELENT,DFGRD0,DFGRD1,NOEL,NPT,LAYER,KSPT,KSTEP,KINC)

C
C      INCLUDE 'ABA_PARAM.INC'
C
C      CHARACTER*8 CMNAME
C
      DIMENSION STRESS(NTENS),STATEV(NSTATV),DDSDDE(NTENS, NTENS),
     1     DDSDDT(6),DRPLDE(6),STRAN(6),DSTRAN(NTENS),
     2     PREDEF(1),DPRED(1),PROPS(NPROPS),COORDS(3),DROT(3,3),
     3     DFGRD0(3,3), DFGRD1(3,3), TIME(2)

      PARAMETER(ZERO=0.D0, ONE=1.D0, TWO=2.D0, THREE=3.D0)

C ----------------------------------------------------------------
C
C     UMAT FOR ISOTROPIC ELASTICITY
C
C     CANNOT BE USED FOR PLANE STRESS
C ----------------------------------------------------------------
C
C      PROPS(1) - E
C
C      PROPS(2) - NU
C ----------------------------------------------------------------
C
C      IF (NDI.NE.3) THEN
C         WRITE (7, *) ’THIS UMAT MAY ONLY BE USED FOR ELEMENTS
C    1        WITH THREE DIRECT STRESS COMPONENTS’
C         CALL XIT
C      ENDIF
C
C     ELASTIC PROPERTIES
      EMOD=1.0D5
      ENU=0.0

      EBULK3=EMOD/(ONE-TWO*ENU)
      EG2=EMOD/(ONE+ENU)
      EG=EG2/TWO
      EG3=THREE*EG
      ELAM=(EBULK3-EG2)/THREE
C
C     ELASTIC STIFFNESS
C
      DO K1=1, NDI
         DO K2=1, NDI
            DDSDDE(K2, K1)=0.0
         END DO
         DDSDDE(K1, K1)=EG2
      END DO

C     XY shear in typical commercial software convention
      DDSDDE(4,4)=1.0D4
C     XZ shear in typical commercial software convention
      DDSDDE(5,5)=3.0D4
C     YZ shear in typical commercial software convention
      DDSDDE(6,6)=2.0D4
C
C     CALCULATE STRESS
C     compute stress from a total strain (using old strain and strain increment)
C
      DO K1=1, NTENS
         STRESS(K1)=0
         DO K2=1, NTENS
            STRESS(K1)=STRESS(K1)+DDSDDE(K2, K1)*(STRAN(K2)+DSTRAN(K2))
         END DO
      END DO
C
      RETURN
      END
