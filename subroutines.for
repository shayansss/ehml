C====================================================================
C
C    This program computes the constitutive behavior of soft tissues
C    by advanced contrained mixture theory and pre-stressing effects.
C    It is particularly developed to be combined with the soil 
C    consolidation theory in Abaqus to encompass the fluid mechanics,
C    along with the pre-stressed poroelasticity, and electrochemically
C    induced osmotic pressure. This code should be run together with
C    pre_stressing.py to get all the required parameters, defining
C    the exact soft tissue model (typically articular cartilage).
C    More information at: https://github.com/shayansss/da.
C
C====================================================================
C
C     SDVINI: sets the initial values of the non-homogeneous
C     material parameters, stored in the state variables.
C
C     Input:
C     STATEV  - State variables.
C     COORDS  - Coordinate information.
C     NSTATV  - Number of state variables.
C     NCRDS   - Number of coordinates.
C     NOEL    - Element number.
C     NPT     - Integration point number.
C     LAYER, KSPT - Unused.
C
C     Output:
C     Updated state variables (STATEV).
C
C     STATEV arguments:
C     1       --->  GAG constant
C     2       --->  Fiber constant
C     3       --->  Solid material constant
C     4       --->  1 for inverse FEA, 0 for others
C     5       --->  GAG pressure
C     15      --->  DET(F)
C     16 - 42 --->  Updated fibrillar directional unit vectors without prestress effects
C     43 - 69 --->  Initial fibrillar directional unit vectors with prestress effects
C     70 - 75 --->  Fibrillar stress components
C     76 - 81 --->  Non-fibrillar stress components
C     82 - 90 --->  Initial DFGRD1
C     91 - 93 --->  Initial coordinates
C     94 - 96 --->  Updated coordinates
C
C     Note:
C     Always set FilLoc to the current directory.
C
      SUBROUTINE SDVINI(STATEV,COORDS,NSTATV,NCRDS,NOEL,NPT,
     1 LAYER,KSPT)
C
      INCLUDE 'ABA_PARAM.INC' ! This enables handling single and double precision with same code
C
      DIMENSION STATEV(NSTATV),COORDS(NCRDS)

      DOUBLE PRECISION DEPTH,RPHI,G(10),RTHETA
      INTEGER i, k0, k1, k2, LenFil
      CHARACTER FilLoc*42, sdvNum*5, NoelTxt*10
C
      PARAMETER (ZERO=0.D0,ONE=1.D0,TWO=2.D0,TEN=10.D0,FOUR=4.D0,
     1 CONS1=5.235987755983D0,PI=3.14159265359D0,FFD=0.57735026919D0)
C
      FilLoc = 'C:\temp\DA\txt\DATA.txt'
      LenFil = LEN_TRIM(FILLOC)
C
      OPEN(UNIT=29,FILE=FilLoc(:LenFil),STATUS='OLD')
       READ(29,*,end=10) k0 ! An identifer (pre-stressing, initial parameter defenition, ...)
10    CLOSE(UNIT=29)
C
      IF (k0.EQ.0) THEN
C
       DO i = 1,NSTATV
        STATEV(i) = 0.01  ! Initialize with a small value to prevent issues with zero material parameters
       ENDDO
C
       STATEV(1) = ZERO  ! Deactivate the prestress for this case.
C
       DO i = 1,NCRDS
        STATEV(90+i) = COORDS(i)
        STATEV(93+i) = COORDS(i)
       ENDDO
C
      ELSEIF (ABS(k0).EQ.1) THEN
C
       DO i = 1,NSTATV
        STATEV(i) = ZERO  ! Initialize with zero.
       ENDDO
C
C   :::::::::: Updating FilLoc with NOEL ::::::::::
C
       LenFil = LenFil - 8
       write(NoelTxt, '(I8)') NOEL  ! converting it to character
       k1 = 0
       k2 = 0
       DO i = 1, LEN(NoelTxt)
        IF (NoelTxt(i:i).NE.' ') THEN
C         write(6,*) NoelTxt(i:i)
         IF (k1.EQ.0) THEN
           k1 = i
         ENDIF
         k2 = i
         LenFil = LenFil + 1
        ENDIF
       ENDDO
       i = LenFil-LEN(NoelTxt(k1:k2))
       FilLoc = FilLoc(:i) // NoelTxt(k1:k2) // '.txt'
C
C       IF (NOEL .EQ. 6097) THEN
C        write(6,*) FilLoc(:LenFil+4)
C       ENDIF
C
C   :::::::::: Now update the SDVs ::::::::::
C
       OPEN(UNIT=28,FILE=FilLoc(:LenFil+4),STATUS='OLD')
        DO WHILE (ABS(k0).EQ.1)
         READ(28,*,end=11) intPoint, STATEV(1), STATEV(2), STATEV(3),
     1    STATEV(16), STATEV(17), STATEV(18), STATEV(19), STATEV(20), 
     2    STATEV(21), STATEV(22), STATEV(23), STATEV(24), STATEV(25), 
     3    STATEV(26), STATEV(27), STATEV(28), STATEV(29), STATEV(30), 
     4    STATEV(31), STATEV(32), STATEV(33), STATEV(34), STATEV(35), 
     5    STATEV(36), STATEV(37), STATEV(38), STATEV(39), STATEV(40), 
     6    STATEV(41), STATEV(42)
         IF (int(intPoint) .EQ. NPT) THEN
          GO TO 11
         ENDIF
        ENDDO
11     CLOSE(UNIT=28)
C
C   :::::::::: Inverse FEA for updating important STATEVs ::::::::::
C
       IF (k0.EQ.-1) THEN
        STATEV(4) = ONE   ! Inverse model
       ELSE
        STATEV(4) = ZERO ! Forward FEA
        STATEV(5)=STATEV(1) ! Initially, GAG pressure and ALPHA1 are equal
       ENDIF
       STATEV(15)=ONE
       DO i = 16,42
        STATEV(27+i) = STATEV(i) ! Initial unit vectors representing the fibrillar directional orientation with prestress effects
       ENDDO
       STATEV(82) = ONE
       STATEV(85) = ONE
       STATEV(88) = ONE
      ENDIF
C
      RETURN
      END
C
C====================================================================
C
C     UMAT: defines the effective solid part of the material.
C
C     Input:
C     STRESS - Cauchy stress.
C     STATEV - State variables.
C     DDSDDE - Jacobian matrix
C     STATEV  - State variables.
C     COORDS  - Coordinate information.
C     NSTATV  - Number of state variables.
C     NCRDS   - Number of coordinates.
C     NOEL    - Element number.
C     NPT     - Integration point number.
C     PROPS   - User-specified material constants in Abaqus CAE.
C     DFGRD0  - Deformation gradient tensor before deformation.
C     DFGRD1  - Deformation gradient tensor after deformation.
C     SSE, SPD, SCD, and the others - Unused.
C
C     Output:
C     Updated STRESS and DDSDDE
C
      SUBROUTINE UMAT(STRESS,STATEV,DDSDDE,SSE,SPD,SCD,
     1 RPL,DDSDDT,DRPLDE,DRPLDT,
     2 STRAN,DSTRAN,TIME,DTIME,TEMP,DTEMP,PREDEF,DPRED,CMNAME,
     3 NDI,NSHR,NTENS,NSTATV,PROPS,NPROPS,COORDS,DROT,PNEWDT,
     4 CELENT,DFGRD0,DFGRD1,NOEL,NPT,LAYER,KSPT,JSTEP,KINC)
C
      INCLUDE 'ABA_PARAM.INC'
C
      CHARACTER*80 CMNAME
      DIMENSION STRESS(NTENS),STATEV(NSTATV),
     1 DDSDDE(NTENS,NTENS),DDSDDT(NTENS),DRPLDE(NTENS),
     2 STRAN(NTENS),DSTRAN(NTENS),TIME(2),PREDEF(1),DPRED(1),
     3 PROPS(NPROPS),COORDS(3),DROT(3,3),DFGRD0(3,3),DFGRD1(3,3),
     4 JSTEP(4)
C
C   :::::::::: Local Variables and Parameters ::::::::::
C
      DOUBLE PRECISION TRANF(3,3),DET,IDENT(3,3),INDEX(2,6),ALPHA1,
     1 ALPHA2,IGAGD,NVEC0(27),RH,W1,W2,W3,W4,W5,W6,STRG(NTENS,NTENS),
     2 EPS,FV1(3),HH,NEWV1(3),NSTR(NTENS),STR,LANDA,DELTAV(NTENS),
     3 BVEC(NTENS),C,NS0,E1MP,E2MP,K1MP,CSTR,DFGRD(3,3),EP,STRS(NTENS),
     4 STATE(NSTATV),GAG,VV(NTENS),DETC,EPSC

      INTEGER i,j,k,l,r,m,n,K1,K2,K3,K4,K5,K6,KKK,FF
      PARAMETER (ZERO=0.D0,ONE=1.D0,TWO=2.D0,THREE=3.D0,FOUR=4.D0,
     1 SEVEN=7.D0,SIX=6.D0,HALF=5.D-1,TEN=10D0)
C
C   :::::::::: Initialization ::::::::::
C
      DO i = 1, 3
       DELTAV(i)=ONE ! Kronecker delta in Voigt notation
       DO j = 1, 3
        IDENT(j,i) = ZERO ! Kronecker delta second-order tensor
        TRANF(j,i) = ZERO ! Transpose of the deformation gradient tensor at the end of the increment.
       ENDDO
       IDENT(i,i) = ONE
      ENDDO
      DO i = 1,NTENS
       STRESS(i)=ZERO ! Cauchy stress tensor that should be updated
       STRS(i)=ZERO ! Fibrillar stress tensor
       DO j = 1,NTENS
        DDSDDE(j,i)=ZERO ! Jacobian matrix tensor that needs updating.
       ENDDO
      ENDDO
C
      IF (STATEV(4).EQ.ONE) THEN
        ALPHA1 = ZERO ! For the inverse model
      ELSE
        ALPHA1=STATEV(1) ! Depth-dependent GAG constant.
      ENDIF
C
      C = 3.009D0  ! Fibrillar relative density constant.
      RH = STATEV(2)  ! Total fibrillar density.
      NS0 = STATEV(3)  ! Depth-dependent solid volume fraction constant.
C
      DO i = 4, NTENS  ! NTENS controls the dimensionality of the code.
        DELTAV(i) = ZERO
      ENDDO
C
      INDEX(1,1)=1 ! Index array is used to avoid assigning equal components due to symmetry of higher-order tensors
      INDEX(2,1)=1
      INDEX(1,2)=2
      INDEX(2,2)=2
      INDEX(1,3)=3
      INDEX(2,3)=3
      INDEX(1,4)=1
      INDEX(2,4)=2
      INDEX(1,5)=1
      INDEX(2,5)=3
      INDEX(1,6)=2
      INDEX(2,6)=3
C
      DO i = 1,27
       NVEC0(i) = STATEV(42+i) ! Initial fibrillar directions
      ENDDO
C
C   :::::::::: Initialization of DDSDDE derivation via perturbation method :::::::::: 
C
      KKK=1
      EP=10D-8
      DO i = 1,3
       DO j = 1,3
        DFGRD(j,i)=DFGRD1(j,i)
       ENDDO
      ENDDO
      DO K6=1,NSTATV
       STATE(K6)=STATEV(K6)
      ENDDO
      i=1
      j=1
80    CONTINUE
      DO K1=1,3
       DO K2=1,3
        DFGRD1(K2,K1)=DFGRD(K2,K1)+(IDENT(K2,i)*DFGRD(j,K1)+IDENT(K2,j)
     1   *DFGRD(i,K1))*EP/TWO
       ENDDO
      ENDDO
90    CONTINUE
      DO K6=1,NTENS
       STRESS(K6)=ZERO
      ENDDO
      DO K6=1,NSTATV
       STATEV(K6)=STATE(K6)
      ENDDO
C
C   :::::::::: Stress calculations ::::::::::
C
      CALL TRANSPOSE(DFGRD1,TRANF)
      CALL VMATMUL(DFGRD1,TRANF,NTENS,BVEC) ! BVEC represents the left Cauchy-Green or Finger deformation tensor
      CALL DETERMINANT(DFGRD1,DET) ! Current volume change from the stress-free state
      HH=NS0*(RH*C)/(TWO*C+SEVEN) ! Contribution of other constants to fibrillar stress
      E1MP=PROPS(1)*HH
      E2MP=PROPS(2)*HH
C       write(6,*) E1MP, E2MP, JSTEP(1)
      DO i=0,8
       IF (i.EQ.2) THEN ! Secondary fibrils have lower density by constant C
        E1MP=E1MP/C
        E2MP=E2MP/C
       ENDIF
       DO m = 1,3
        FV1(m)=ZERO ! FV vector is the inner product of DFGRD1 and NVEC0.
        DO n = 1,3
         FV1(m)=DFGRD1(m,n)*NVEC0(3*i+n)+FV1(m)
        ENDDO
       ENDDO
       LANDA=SQRT(FV1(1)**TWO+FV1(2)**TWO+FV1(3)**TWO) ! elongation
       EPS=LOG(LANDA) ! The fibril logarithmic strain
       DO n=1,3
        NEWV1(n)=FV1(n)/LANDA ! NEWV1 represents the current fibril direction
        STATEV(n+3*i+15)=NEWV1(n) ! Degrees of new directions
       ENDDO
       IF (EPS.GT.ZERO) THEN
        STR=(E1MP+E2MP*EPS)*EPS*LANDA/DET ! The local fibril stress
        DO K6=1,NTENS
         K3=INDEX(1,K6)
         K4=INDEX(2,K6)
         VV(K6)=NEWV1(K3)*NEWV1(K4) ! VV is the dyadic product of current direction vectors, representing the structural vector
         STRS(K6)=STR*VV(K6) ! STRS represents the global fibril stress
         STRESS(K6)=STRESS(K6)+STRS(K6)
        ENDDO
       ENDIF
      ENDDO
      DO K1=1,NTENS
        STATEV(69+K1)=STRESS(K1) ! Stress in the fibrillar part
      ENDDO
C
      GM=0.723D0 ! Neo-Hookean constant for the PG material
C      GM=72.3D0
      GM=GM*NS0*(ONE-RH) ! Contribution of other constants
      W5=GM/DET
      W6=((LOG(DET)/SIX)*(((THREE*NS0/(DET-NS0))
     1 *((DET*LOG(DET)/(DET-NS0))-TWO))-FOUR)+(DET**(TWO/THREE)))*W5
      DO K6=1,NTENS
       NSTR(K6)=-DELTAV(K6)*W6+BVEC(K6)*W5
      ENDDO
      DO K1=1,NTENS
        STATEV(75+K1)=NSTR(K1) ! Stress in the non-fibrillar part
      ENDDO
C
      IF (PROPS(3).EQ.1) THEN
       IF (JSTEP(1).EQ.1) THEN
        STATEV(15)=DET ! DET(F) with RESPE
        ALPHA2=ZERO ! GAG constant
        DETC=ONE
       ELSE
        DETC=DET/STATEV(15) ! Current volume change from the pre-stressed state
        ALPHA2=PROPS(4)
       ENDIF
       GAG=ALPHA1*(DETC**(-ALPHA2))
      ELSE
       ALPHA2=PROPS(4)
       GAG=ALPHA1*(DET**(-ALPHA2))-ALPHA1
      ENDIF
C
      STATEV(5)=GAG ! S22 stress of the GAG part
      DO K6=1,3
       STRESS(K6)=NSTR(K6)-GAG+STRESS(K6)
      ENDDO
      DO K6=4,NTENS
       STRESS(K6)=NSTR(K6)+STRESS(K6)
      ENDDO
C
C   :::::::::: The 2nd part of DDSDDE derivation via perturbation method ::::::::::
C
      IF (KKK.LT.NTENS) THEN
       DO K5=1,NTENS
        DDSDDE(K5,KKK)=STRESS(K5)*DET
       ENDDO
       KKK=KKK+1
       i=INDEX(1,KKK)
       j=INDEX(2,KKK)
       GO TO 80
      ENDIF
      IF (KKK.EQ.NTENS) THEN
       DO K5=1,NTENS
        DDSDDE(K5,KKK)=STRESS(K5)*DET
       ENDDO
       DO K1=1,3
        DO K2=1,3
         DFGRD1(K2,K1)=DFGRD(K2,K1)
        ENDDO
       ENDDO
       KKK=KKK+1
       GO TO 90
      ENDIF
      W2=ONE/EP
      W1=W2/DET
      DO K6=1,NTENS
       DO K5=1,NTENS
        DDSDDE(K5,K6)=W1*DDSDDE(K5,K6)-W2*STRESS(K5)
       ENDDO
      ENDDO
C
C   :::::::::: Updates to other state variables :::::::::: 
C
      STATEV(2)=RH/DET
      STATEV(3)=NS0/DET
C
C
95    RETURN
      END
C
C====================================================================
C
C     TRANSPOSE: Computes the transpose of a 3x3 matrix
C
C     Input:
C     A - 3D matrix to be transposed
C
C     Output:
C     AT - Transposed matrix
C
      SUBROUTINE TRANSPOSE(A,AT)
      INTEGER i , j
      DOUBLE PRECISION A(3,3), AT(3,3)
      Do i = 1 , 3
       DO j = 1 , 3
        AT(j,i) = A(i,j)
       ENDDO
      ENDDO
      RETURN
      END
C
C====================================================================
C
C     DETERMINANT: Computes the determinant of a 3D matrix
C
C     Input:
C     A - 3D matrix
C
C     Output:
C     DET - Determinant of matrix A
C
      SUBROUTINE DETERMINANT(A,DET)
      DOUBLE PRECISION A(3,3), DET
      DET=A(1,1)*A(2,2)*A(3,3)-A(1,1)*A(2,3)*A(3,2)
     1 -A(2,1)*A(1,2)*A(3,3)+A(2,1)*A(1,3)*A(3,2)
     2 +A(3,1)*A(1,2)*A(2,3)-A(3,1)*A(1,3)*A(2,2)
      RETURN
      END
C
C====================================================================
C
C     VMATMUL: Computes the inner product of two 2D or 3D matrices 
C     and stores the result in Voigt notation, assuming symmetric outputs.
C
C     Input:
C     A - First 2D or 3D matrix
C     B - Second 2D or 3D matrix
C     N - Size of resulting array (4 for symmetric 2D, 6 for symmetric 3D problems)
C
C     Output:
C     C - Resulting product in Voigt notation
C
      SUBROUTINE VMATMUL(A,B,N,C)
      DOUBLE PRECISION A(3,3), B(3,3), C(N)
      C(1)=A(1,1)*B(1,1)+A(1,2)*B(2,1)+A(1,3)*B(3,1)
      C(2)=A(2,1)*B(1,2)+A(2,2)*B(2,2)+A(2,3)*B(3,2)
      C(3)=A(3,1)*B(1,3)+A(3,2)*B(2,3)+A(3,3)*B(3,3)
      C(4)=A(1,1)*B(1,2)+A(1,2)*B(2,2)+A(1,3)*B(3,2)
      IF (N.EQ.6) THEN
        C(5)=A(1,1)*B(1,3)+A(1,2)*B(2,3)+A(1,3)*B(3,3)
        C(6)=A(2,1)*B(1,3)+A(2,2)*B(2,3)+A(2,3)*B(3,3)
      ENDIF
      RETURN
      END
C
C====================================================================
C