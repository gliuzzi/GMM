      SUBROUTINE GROUP ( GVALUE, LGVALU, FVALUE, GPVALU, NCALCG, 
     *                   ITYPEG, ISTGPA, ICALCG, LTYPEG, LSTGPA, 
     *                   LCALCG, LFVALU, LGPVLU, DERIVS, IGSTAT )
      INTEGER LGVALU, NCALCG, LTYPEG, LSTGPA
      INTEGER LCALCG, LFVALU, LGPVLU, IGSTAT
      LOGICAL DERIVS
      INTEGER ITYPEG(LTYPEG), ISTGPA(LSTGPA), ICALCG(LCALCG)
      DOUBLE PRECISION GVALUE(LGVALU,3), FVALUE(LFVALU), GPVALU(LGPVLU)
C
C  Problem name : VAREIGVL  
C
C  -- produced by SIFdecode 1.0
C
      INTEGER IGRTYP, IGROUP, IPSTRT, JCALCG
      INTEGER IPOWER, PM1   
      DOUBLE PRECISION GVAR  , POWER 
      IGSTAT = 0
      DO     3 JCALCG = 1, NCALCG
       IGROUP = ICALCG(JCALCG)
       IGRTYP = ITYPEG(IGROUP)
       IF ( IGRTYP == 0 ) GO TO     3
       IPSTRT = ISTGPA(IGROUP) - 1
       GO TO (    1,    2
     *                                                        ), IGRTYP
C
C  Group type : LQ      
C
    1  CONTINUE 
       GVAR  = FVALUE(IGROUP)
       POWER = GPVALU(IPSTRT+     1)
       IPOWER= POWER                                    
       PM1   = IPOWER - 1                               
       IF ( .NOT. DERIVS ) THEN
        GVALUE(IGROUP,1)= GVAR**IPOWER / POWER                     
       ELSE
        GVALUE(IGROUP,2)= GVAR**PM1                                
        GVALUE(IGROUP,3)= PM1 * GVAR**( IPOWER - 2)                
       END IF
       GO TO     3
C
C  Group type : LQ2     
C
    2  CONTINUE 
       GVAR  = FVALUE(IGROUP)
       POWER = GPVALU(IPSTRT+     1)
       IF ( .NOT. DERIVS ) THEN
        GVALUE(IGROUP,1)= GVAR ** POWER / POWER                    
       ELSE
        GVALUE(IGROUP,2)= GVAR ** (POWER - 1.0D0)                  
        GVALUE(IGROUP,3)= (POWER - 1.0D0) * GVAR ** (POWER - 2.0D0)
       END IF
    3 CONTINUE
      RETURN
      END
