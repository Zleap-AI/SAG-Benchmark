/* BM25 calculations adapted from SQLite ext/fts5/fts5_aux.c (public domain).
 * Reuse exact per-query statistics across filtered searches.
 */
#include <sqlite3ext.h>
SQLITE_EXTENSION_INIT1
#include <math.h>
#include <string.h>
#include <assert.h>
typedef struct Fts5Bm25Data Fts5Bm25Data;
struct Fts5Bm25Data {
  int nPhrase;                    /* Number of phrases in query */
  double avgdl;                   /* Average number of tokens in each row */
  double *aIDF;                   /* IDF for each phrase */
  double *aFreq;                  /* Array used to calculate phrase freq. */
};

/*
** Callback used by fts5Bm25GetData() to count the number of rows in the
** table matched by each individual phrase within the query.
*/
static int fts5CountCb(
  const Fts5ExtensionApi *pApi, 
  Fts5Context *pFts,
  void *pUserData                 /* Pointer to sqlite3_int64 variable */
){
  sqlite3_int64 *pn = (sqlite3_int64*)pUserData;
  (void)pApi; (void)pFts;
  (*pn)++;
  return SQLITE_OK;
}

/*
** Set *ppData to point to the Fts5Bm25Data object for the current query. 
** If the object has not already been allocated, allocate and populate it
** now.
*/
static int fts5Bm25GetData(
  const Fts5ExtensionApi *pApi, 
  Fts5Context *pFts,
  Fts5Bm25Data **ppData           /* OUT: bm25-data object for this query */
){
  int rc = SQLITE_OK;             /* Return code */
  Fts5Bm25Data *p;                /* Object to return */

  p = (Fts5Bm25Data*)pApi->xGetAuxdata(pFts, 0);
  if( p==0 ){
    int nPhrase;                  /* Number of phrases in query */
    sqlite3_int64 nRow = 0;       /* Number of rows in table */
    sqlite3_int64 nToken = 0;     /* Number of tokens in table */
    sqlite3_int64 nByte;          /* Bytes of space to allocate */
    int i;

    /* Allocate the Fts5Bm25Data object */
    nPhrase = pApi->xPhraseCount(pFts);
    nByte = sizeof(Fts5Bm25Data) + nPhrase*2*sizeof(double);
    p = (Fts5Bm25Data*)sqlite3_malloc64(nByte);
    if( p==0 ){
      rc = SQLITE_NOMEM;
    }else{
      memset(p, 0, (size_t)nByte);
      p->nPhrase = nPhrase;
      p->aIDF = (double*)&p[1];
      p->aFreq = &p->aIDF[nPhrase];
    }

    /* Calculate the average document length for this FTS5 table */
    if( rc==SQLITE_OK ) rc = pApi->xRowCount(pFts, &nRow);
    assert( rc!=SQLITE_OK || nRow>0 );
    if( rc==SQLITE_OK ) rc = pApi->xColumnTotalSize(pFts, -1, &nToken);
    if( rc==SQLITE_OK ) p->avgdl = (double)nToken  / (double)nRow;

    /* Calculate an IDF for each phrase in the query */
    for(i=0; rc==SQLITE_OK && i<nPhrase; i++){
      sqlite3_int64 nHit = 0;
      rc = pApi->xQueryPhrase(pFts, i, (void*)&nHit, fts5CountCb);
      if( rc==SQLITE_OK ){
        /* Calculate the IDF (Inverse Document Frequency) for phrase i.
        ** This is done using the standard BM25 formula as found on wikipedia:
        **
        **   IDF = log( (N - nHit + 0.5) / (nHit + 0.5) )
        **
        ** where "N" is the total number of documents in the set and nHit
        ** is the number that contain at least one instance of the phrase
        ** under consideration.
        **
        ** The problem with this is that if (N < 2*nHit), the IDF is 
        ** negative. Which is undesirable. So the minimum allowable IDF is
        ** (1e-6) - roughly the same as a term that appears in just over
        ** half of set of 5,000,000 documents.  */
        double idf = log( (nRow - nHit + 0.5) / (nHit + 0.5) );
        if( idf<=0.0 ) idf = 1e-6;
        p->aIDF[i] = idf;
      }
    }

    if( rc!=SQLITE_OK ){
      sqlite3_free(p);
    }else{
      rc = pApi->xSetAuxdata(pFts, p, sqlite3_free);
    }
    if( rc!=SQLITE_OK ) p = 0;
  }
  *ppData = p;
  return rc;
}


static void rankStats(const Fts5ExtensionApi *api, Fts5Context *fts, sqlite3_context *ctx, int argc, sqlite3_value **argv){
  (void)argc; (void)argv;
  Fts5Bm25Data *p;
  int rc=fts5Bm25GetData(api,fts,&p);
  if(rc!=SQLITE_OK){sqlite3_result_error_code(ctx,rc);return;}
  int size=(p->nPhrase+1)*sizeof(double);
  double *out=sqlite3_malloc(size);
  if(!out){sqlite3_result_error_nomem(ctx);return;}
  out[0]=p->avgdl;
  memcpy(out+1,p->aIDF,p->nPhrase*sizeof(double));
  sqlite3_result_blob(ctx,out,size,sqlite3_free);
}
static int statsFromBlob(const Fts5ExtensionApi *api,Fts5Context *fts,sqlite3_value *value,Fts5Bm25Data **out){
  Fts5Bm25Data *p=api->xGetAuxdata(fts,0);
  if(!p){
    int bytes=sqlite3_value_bytes(value);
    if(bytes<(int)sizeof(double) || bytes%(int)sizeof(double))return SQLITE_MISMATCH;
    int n=bytes/sizeof(double)-1;
    if(n>api->xPhraseCount(fts))return SQLITE_MISMATCH;
    const double *blob=sqlite3_value_blob(value);
    p=sqlite3_malloc64(sizeof(*p)+n*2*sizeof(double));
    if(!p)return SQLITE_NOMEM;
    p->nPhrase=n;p->avgdl=blob[0];p->aIDF=(double*)&p[1];p->aFreq=p->aIDF+n;
    memcpy(p->aIDF,blob+1,n*sizeof(double));
    int rc=api->xSetAuxdata(fts,p,sqlite3_free);
    if(rc!=SQLITE_OK)return rc;
  }
  *out=p;return SQLITE_OK;
}
static void cachedBm25(
  const Fts5ExtensionApi *pApi,   /* API offered by current FTS version */
  Fts5Context *pFts,              /* First arg to pass to pApi functions */
  sqlite3_context *pCtx,          /* Context for returning result/error */
  int nVal,                       /* Number of values in apVal[] array */
  sqlite3_value **apVal           /* Array of trailing arguments */
){
  const double k1 = 1.2;          /* Constant "k1" from BM25 formula */
  const double b = 0.75;          /* Constant "b" from BM25 formula */
  int rc;                         /* Error code */
  double score = 0.0;             /* SQL function return value */
  Fts5Bm25Data *pData;            /* Values allocated/calculated once only */
  int i;                          /* Iterator variable */
  int nInst = 0;                  /* Value returned by xInstCount() */
  double D = 0.0;                 /* Total number of tokens in row */
  double *aFreq = 0;              /* Array of phrase freq. for current row */

  /* Calculate the phrase frequency (symbol "f(qi,D)" in the documentation)
  ** for each phrase in the query for the current row. */
  rc = statsFromBlob(pApi, pFts, apVal[0], &pData);
  if( rc==SQLITE_OK ){
    aFreq = pData->aFreq;
    memset(aFreq, 0, sizeof(double) * pData->nPhrase);
    rc = pApi->xInstCount(pFts, &nInst);
  }
  for(i=0; rc==SQLITE_OK && i<nInst; i++){
    int ip; int ic; int io;
    rc = pApi->xInst(pFts, i, &ip, &ic, &io);
    if( rc==SQLITE_OK ){
      double w = (nVal > ic+1) ? sqlite3_value_double(apVal[ic+1]) : 1.0;
      if(ip < pData->nPhrase) aFreq[ip] += w;
    }
  }

  /* Figure out the total size of the current row in tokens. */
  if( rc==SQLITE_OK ){
    int nTok;
    rc = pApi->xColumnSize(pFts, -1, &nTok);
    D = (double)nTok;
  }

  /* Determine and return the BM25 score for the current row. Or, if an
  ** error has occurred, throw an exception. */
  if( rc==SQLITE_OK ){
    for(i=0; i<pData->nPhrase; i++){
      score += pData->aIDF[i] * (
          ( aFreq[i] * (k1 + 1.0) ) / 
          ( aFreq[i] + k1 * (1 - b + b * D / pData->avgdl) )
      );
    }
    sqlite3_result_double(pCtx, -1.0 * score);
  }else{
    sqlite3_result_error_code(pCtx, rc);
  }
}


#if defined(SAG_STATIC)
int sag_register_rank(sqlite3 *db){
#else
#if defined(_WIN32)
__declspec(dllexport)
#endif
int sqlite3_extension_init(sqlite3 *db,char **error,const sqlite3_api_routines *routines){
  (void)error;
  SQLITE_EXTENSION_INIT2(routines);
#endif
  fts5_api *api=0;sqlite3_stmt *stmt=0;
  int rc=sqlite3_prepare_v2(db,"SELECT fts5(?1)",-1,&stmt,0);
  if(rc!=SQLITE_OK)return rc;
  sqlite3_bind_pointer(stmt,1,&api,"fts5_api_ptr",0);
  sqlite3_step(stmt);sqlite3_finalize(stmt);
  if(!api)return SQLITE_ERROR;
  rc=api->xCreateFunction(api,"sag_stats",0,rankStats,0);
  if(rc!=SQLITE_OK)return rc;
  return api->xCreateFunction(api,"sag_bm25",0,cachedBm25,0);
}
