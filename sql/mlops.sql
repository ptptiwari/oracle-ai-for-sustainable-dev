-- USER SQL
CREATE USER OMLOPSUSER IDENTIFIED BY Welcome12345;

-- ADD ROLES
GRANT CONNECT TO OMLOPSUSER;
GRANT CONSOLE_DEVELOPER TO OMLOPSUSER;
GRANT DWROLE TO OMLOPSUSER;
GRANT GRAPH_DEVELOPER TO OMLOPSUSER;
GRANT OML_DEVELOPER TO OMLOPSUSER;
GRANT RESOURCE TO OMLOPSUSER;
ALTER USER OMLOPSUSER DEFAULT ROLE CONSOLE_DEVELOPER,DWROLE,GRAPH_DEVELOPER,OML_DEVELOPER;

-- REST ENABLE
BEGIN
    ORDS_ADMIN.ENABLE_SCHEMA(
        p_enabled => TRUE,
        p_schema => 'OMLOPSUSER',
        p_url_mapping_type => 'BASE_PATH',
        p_url_mapping_pattern => 'omlopsuser',
        p_auto_rest_auth=> TRUE
    );
    -- ENABLE DATA SHARING
    C##ADP$SERVICE.DBMS_SHARE.ENABLE_SCHEMA(
            SCHEMA_NAME => 'OMLOPSUSER',
            ENABLED => TRUE
    );
    commit;
END;
/

-- ENABLE GRAPH
ALTER USER OMLOPSUSER GRANT CONNECT THROUGH GRAPH$PROXY_USER;

-- ENABLE OML
ALTER USER OMLOPSUSER GRANT CONNECT THROUGH OML$PROXY;

