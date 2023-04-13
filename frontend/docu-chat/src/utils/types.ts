export interface FileUploadVariables {
  base64FileString: string;
  filename: string;
  customMeta: Record<string, any>;
  description: string;
  title: string;
}

export interface FileDetailsProps {
    title?: string;
    description?: string;
  }
  
export interface FileUploadPackageProps {
file: File;
formData: FileDetailsProps;
}

export interface RightColProps {
files: FileUploadPackageProps[];
selected_file_num: number;
selected_doc: number;
handleChange: (a: any) => void;
}

export enum UploadStatus {
    NOT_STARTED = "NOT_STARTED",
    SUCCESS = "SUCCESS",
    FAILED = "FAILED",
    UPLOADING = "UPLOADING"
}