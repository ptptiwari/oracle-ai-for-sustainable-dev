create or replace FUNCTION call_analyze_audio_api_objectstore (
    p_endpoint VARCHAR2,
    p_compartment_ocid VARCHAR2,
    p_namespaceName VARCHAR2,
    p_bucketName VARCHAR2,
    p_objectName VARCHAR2,
    p_featureType VARCHAR2,
    p_label VARCHAR2
) RETURN CLOB IS
    resp DBMS_CLOUD_TYPES.resp;
    json_response CLOB;
BEGIN
    resp := DBMS_CLOUD.send_request(
        credential_name => 'OCI_KEY_CRED',
        uri => p_endpoint || '/20220125/actions/analyzeImage',
        method => 'POST',
        body => UTL_RAW.cast_to_raw(
            JSON_OBJECT(
                'features' VALUE JSON_ARRAY(
                    JSON_OBJECT('featureType' VALUE p_featureType)
                ),
                    'image' VALUE JSON_OBJECT(
                    'source' VALUE 'OBJECT_STORAGE',
                    'namespaceName' VALUE p_namespaceName,
                    'bucketName' VALUE p_bucketName,
                    'objectName' VALUE p_objectName
                ),
                'compartmentId' VALUE p_compartment_ocid
            )
        )
    );

    json_response := DBMS_CLOUD.get_response_text(resp);
    dbms_output.put_line('json_response: ' || json_response);
    INSERT INTO aispeech_results VALUES (SYS_GUID(), SYSTIMESTAMP, p_label, json_response );
    RETURN json_response;
EXCEPTION
    WHEN OTHERS THEN
        -- Handle exceptions if needed and return an error message or raise
        RAISE;
END call_analyze_audio_api_objectstore;
/

BEGIN
    ORDS.ENABLE_OBJECT(
        P_ENABLED      => TRUE,
        P_SCHEMA      => 'AIUSER',
        P_OBJECT      =>  'CALL_ANALYZE_AUDIO_API_OBJECTSTORE',
        P_OBJECT_TYPE      => 'FUNCTION',
        P_OBJECT_ALIAS      => 'call_analyze_audio_api_objectstore',
        P_AUTO_REST_AUTH      => FALSE
    );
    COMMIT;
END;