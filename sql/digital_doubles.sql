CREATE TABLE digital_double_data (
    id NUMBER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    participant_firstname VARCHAR2(100),
    participant_lastname VARCHAR2(100),
    participant_email VARCHAR2(100) NOT NULL,
    participant_company VARCHAR2(100),
    participant_role VARCHAR2(100),
    participant_tshirt VARCHAR2(100),
    participant_comments VARCHAR2(200),
    id_image_in BLOB,
    image_in BLOB,
    video_in BLOB,
    modelglburl_out VARCHAR2(1000),
    modelfbxurl_out VARCHAR2(1000),
    modelusdzurl_out VARCHAR2(1000),
    thumbnailurl_out VARCHAR2(1000),
    videourl_out BLOB,
    video_out BLOB,
    similar_image_out BLOB
);
/


CREATE OR REPLACE PROCEDURE insert_digital_double_data (
    p_participant_firstname IN VARCHAR2,
    p_participant_lastname IN VARCHAR2,
    p_participant_email IN VARCHAR2,
    p_participant_company IN VARCHAR2,
    p_participant_role IN VARCHAR2,
    p_participant_tshirt IN VARCHAR2,
    p_participant_comments IN VARCHAR2,
    p_id_image_in IN BLOB,
    p_image_in IN BLOB,
    p_video_in IN BLOB
) IS
BEGIN
    INSERT INTO digital_double_data (
        participant_firstname,
        participant_lastname,
        participant_email,
        participant_company,
        participant_role,
        participant_tshirt,
        participant_comments,
        id_image_in,
        image_in,
        video_in
    )
    VALUES (
        p_participant_firstname,
        p_participant_lastname,
        p_participant_email,
        p_participant_company,
        p_participant_role,
        p_participant_tshirt,
        p_participant_comments,
        p_id_image_in,
        p_image_in,
        p_video_in
    );
END insert_digital_double_data;
/

CREATE OR REPLACE PROCEDURE update_digital_double_data (
    p_participant_email IN VARCHAR2,
    p_modelglburl_out IN VARCHAR2,
    p_modelfbxurl_out IN VARCHAR2,
    p_modelusdzurl_out IN VARCHAR2,
    p_thumbnailurl_out IN VARCHAR2,
    p_videourl_out IN BLOB,
    p_video_out IN BLOB,
    p_similar_image_out IN BLOB
) IS
BEGIN
    UPDATE digital_double_data
    SET modelglburl_out = p_modelglburl_out,
        modelfbxurl_out = p_modelfbxurl_out,
        modelusdzurl_out = p_modelusdzurl_out,
        thumbnailurl_out = p_thumbnailurl_out,
        videourl_out = p_videourl_out,
        video_out = p_video_out,
        similar_image_out = p_similar_image_out
    WHERE participant_email = p_participant_email;
END update_digital_double_data;
/

CREATE OR REPLACE PROCEDURE get_digital_double_data (
    p_participant_email IN VARCHAR2,
    p_participant_firstname OUT VARCHAR2,
    p_participant_lastname OUT VARCHAR2,
    p_participant_company OUT VARCHAR2,
    p_participant_role OUT VARCHAR2,
    p_participant_tshirt OUT VARCHAR2,
    p_participant_comments OUT VARCHAR2,
    p_id_image_in OUT BLOB,
    p_image_in OUT BLOB,
    p_video_in OUT BLOB,
    p_modelglburl_out OUT VARCHAR2,
    p_modelfbxurl_out OUT VARCHAR2,
    p_modelusdzurl_out OUT VARCHAR2,
    p_thumbnailurl_out OUT VARCHAR2,
    p_videourl_out OUT BLOB,
    p_video_out OUT BLOB,
    p_similar_image_out OUT BLOB
) IS
BEGIN
    SELECT participant_firstname, participant_lastname, participant_company,
           participant_role, participant_tshirt, participant_comments,
           id_image_in, image_in, video_in,
           modelglburl_out, modelfbxurl_out, modelusdzurl_out,
           thumbnailurl_out, videourl_out, video_out, similar_image_out
    INTO p_participant_firstname, p_participant_lastname, p_participant_company,
         p_participant_role, p_participant_tshirt, p_participant_comments,
         p_id_image_in, p_image_in, p_video_in,
         p_modelglburl_out, p_modelfbxurl_out, p_modelusdzurl_out,
         p_thumbnailurl_out, p_videourl_out, p_video_out, p_similar_image_out
    FROM digital_double_data
    WHERE participant_email = p_participant_email;
END get_digital_double_data;
/



--ORDS...


BEGIN
    ORDS.ENABLE_OBJECT(
        P_ENABLED      => TRUE,
        P_SCHEMA      => 'OMLOPSUSER',
        P_OBJECT      =>  'INSERT_DIGITAL_DOUBLE_DATA',
        P_OBJECT_TYPE      => 'PROCEDURE',
        P_OBJECT_ALIAS      => 'insert_digital_double_data',
        P_AUTO_REST_AUTH      => FALSE
    );
    COMMIT;
END;
/

BEGIN
    ORDS.ENABLE_OBJECT(
        P_ENABLED      => TRUE,
        P_SCHEMA      => 'OMLOPSUSER',
        P_OBJECT      =>  'GET_DIGITAL_DOUBLE_DATA',
        P_OBJECT_TYPE      => 'PROCEDURE',
        P_OBJECT_ALIAS      => 'get_digital_double_data',
        P_AUTO_REST_AUTH      => FALSE
    );
    COMMIT;
END;
/

BEGIN
    ORDS.ENABLE_OBJECT(
        P_ENABLED      => TRUE,
        P_SCHEMA      => 'OMLOPSUSER',
        P_OBJECT      =>  'UPDATE_DIGITAL_DOUBLE_DATA',
        P_OBJECT_TYPE      => 'PROCEDURE',
        P_OBJECT_ALIAS      => 'update_digital_double_data',
        P_AUTO_REST_AUTH      => FALSE
    );
    COMMIT;
END;
/

